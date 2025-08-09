"""
Adaptive Photo Mosaic GUI

Features:
- Adaptive (quadtree) tiling based on detail/variance (smaller tiles in detailed areas)
- Matches each tile to the closest source image by average RGB color
- Option to output full mosaic or half/half (original | mosaic) with adjustable split
- Simple GUI to pick target image, source folder, output folder and tweak a few parameters
- Adjustable opacity blending, flexible per-axis stretching and smart merging of adjacent similar-source tiles into single spanning images
- Preview group overlay, diagonal/gap adjacency, hash-based clustering of very similar source images, min-group thresholds

Dependencies: Pillow, numpy
Install: pip install pillow numpy
Run: python adaptive_photo_mosaic_gui.py

Notes:
- For large source libraries this may take some time; the script computes average colors and a small perceptual hash for matching and clustering.
- The code is intentionally readable so you can tweak thresholds / tile sizes / stretch limits.
"""

import os
import threading
import time
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np

# ---------- Utilities ----------
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}


def is_image_file(path):
    return os.path.splitext(path.lower())[1] in IMAGE_EXTS


def compute_average_color_pil(img, small=32):
    """Compute average color of a PIL image quickly by downscaling first."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Downsize for speed
    w, h = img.size
    if max(w, h) > small:
        img = img.resize((small, int(small * h / w)) if w > h else (int(small * w / h), small), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return np.array([0.0, 0.0, 0.0])
    return arr.mean(axis=(0, 1))


def average_hash(img, hash_size=8):
    """Compute a simple average hash (ahash) returned as integer (hash_size*hash_size bits).
    Small and fast; good enough to cluster visually-similar images.
    """
    if img.mode != 'L':
        im = img.convert('L')
    else:
        im = img
    im = im.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    # pack into integer
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming_distance(a, b):
    return (a ^ b).bit_count()


# ---------- Quadtree tiling ----------


def quadtree_tiles(np_img, min_tile=24, max_tile=256, var_threshold=500.0):
    """Return a list of tiles (x,y,w,h) adaptively split by variance.
    Rules:
      - If w or h <= min_tile -> stop
      - If both w and h > max_tile -> force split
      - Else split if variance > var_threshold
    """
    h, w = np_img.shape[:2]
    tiles = []

    def _region_var(x, y, ww, hh):
        reg = np_img[y:y+hh, x:x+ww]
        # variance per channel averaged
        v = reg.var(axis=(0, 1)).mean()
        return float(v)

    def split(x, y, ww, hh):
        v = _region_var(x, y, ww, hh)
        if ww <= min_tile or hh <= min_tile:
            tiles.append((x, y, ww, hh))
            return
        if (ww > max_tile and hh > max_tile) or (v > var_threshold and ww > min_tile and hh > min_tile):
            w2 = ww // 2
            h2 = hh // 2
            if w2 < 1 or h2 < 1:
                tiles.append((x, y, ww, hh))
                return
            split(x, y, w2, h2)
            split(x + w2, y, ww - w2, h2)
            split(x, y + h2, w2, hh - h2)
            split(x + w2, y + h2, ww - w2, hh - h2)
        else:
            tiles.append((x, y, ww, hh))

    split(0, 0, w, h)
    return tiles


# ---------- Mosaic construction ----------


def build_source_index(folder, thumb_for_avg=32, hash_size=8, max_images=None, progress_callback=None):
    """Scan folder, compute average color and a small perceptual hash for each image and return list of (path, avg_color, ahash)
    Does not keep opened PIL images in memory - they are opened later when pasted.
    """
    items = []
    all_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if is_image_file(f):
                all_files.append(os.path.join(root, f))
    if max_images:
        all_files = all_files[:max_images]
    n = len(all_files)
    for i, p in enumerate(all_files):
        try:
            with Image.open(p) as img:
                avg = compute_average_color_pil(img, small=thumb_for_avg)
                ah = average_hash(img, hash_size=hash_size)
            items.append((p, avg, ah))
        except Exception as e:
            # skip broken images silently
            print('Skipping', p, 'error:', e)
        if progress_callback and i % 10 == 0:
            progress_callback(i, n)
    if progress_callback:
        progress_callback(n, n)
    return items


def _render_source_for_region(src_path, out_w, out_h, tile_avg, stretch_mode, max_stretch_factor, force_full_stretch):
    """Load src_path and render a PIL image sized out_w x out_h according to stretch rules.
    Returns an Image object.
    """
    try:
        with Image.open(src_path) as simg:
            s = simg.convert('RGB')
            sw, sh = s.size
            target_aspect = out_w / out_h if out_h != 0 else 1.0
            src_aspect = sw / sh if sh != 0 else 1.0

            if stretch_mode == 'keep':
                # center-crop then resize
                if src_aspect > target_aspect:
                    new_w = int(sh * target_aspect)
                    left = max(0, (sw - new_w) // 2)
                    s = s.crop((left, 0, left + new_w, sh))
                else:
                    new_h = int(sw / target_aspect) if target_aspect != 0 else sh
                    top = max(0, (sh - new_h) // 2)
                    s = s.crop((0, top, sw, top + new_h))
                s = s.resize((out_w, out_h), Image.Resampling.LANCZOS)
                return s

            elif stretch_mode == 'fit':
                # fit inside, preserve aspect, letterbox on avg-color bg
                if src_aspect > target_aspect:
                    new_w = out_w
                    new_h = max(1, int(round(new_w / src_aspect)))
                else:
                    new_h = out_h
                    new_w = max(1, int(round(new_h * src_aspect)))
                s = s.resize((new_w, new_h), Image.Resampling.LANCZOS)
                bg = Image.new('RGB', (out_w, out_h), tuple([int(max(0, min(255, c))) for c in tile_avg]))
                offx = (out_w - new_w) // 2
                offy = (out_h - new_h) // 2
                bg.paste(s, (offx, offy))
                return bg

            else:  # 'stretch' (per-axis)
                sx = out_w / sw if sw != 0 else 1.0
                sy = out_h / sh if sh != 0 else 1.0

                if sx <= max_stretch_factor and sy <= max_stretch_factor:
                    # allowed full stretch
                    s = s.resize((out_w, out_h), Image.Resampling.LANCZOS)
                    return s

                else:
                    if force_full_stretch:
                        s = s.resize((out_w, out_h), Image.Resampling.LANCZOS)
                        return s
                    else:
                        sx_l = min(sx, max_stretch_factor)
                        sy_l = min(sy, max_stretch_factor)
                        new_w = max(1, int(round(sw * sx_l)))
                        new_h = max(1, int(round(sh * sy_l)))
                        s = s.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        bg = Image.new('RGB', (out_w, out_h), tuple([int(max(0, min(255, c))) for c in tile_avg]))
                        offx = (out_w - new_w) // 2
                        offy = (out_h - new_h) // 2
                        bg.paste(s, (offx, offy))
                        return bg
    except Exception:
        # fallback solid color
        fill = tuple([int(max(0, min(255, v))) for v in tile_avg])
        return Image.new('RGB', (out_w, out_h), fill)


def create_mosaic_from_tiles(target_img, tiles, source_index, progress_callback=None, stop_flag=None,
                             stretch_mode='keep', max_stretch_factor=3.0, force_full_stretch=False,
                             enable_merge_groups=True, merge_area_threshold=0.85,
                             hash_threshold=6, allow_diagonal=False, gap_tolerance=2,
                             min_group_tiles=3, min_merge_bbox_area=5000,
                             preview_only=False):
    """Create a mosaic PIL image the same size as target_img.
    - Clusters source images by perceptual hash with the given hash_threshold (hamming distance).
    - Groups adjacent tiles that map to the same cluster (not just identical file) and merges large rectangular groups.
    - If preview_only=True the function will return (mosaic_placeholder, groups_info) where groups_info
      is a list of (min_x,min_y,bbox_w,bbox_h,coverage,cluster_id,num_tiles) so the GUI can render an overlay preview.
    """
    if target_img.mode != 'RGB':
        target_img = target_img.convert('RGB')
    w, h = target_img.size
    np_target = np.asarray(target_img, dtype=np.float32)

    # Extract arrays from source_index (path, avg, ahash)
    if len(source_index) == 0:
        src_colors = np.zeros((0, 3), dtype=np.float32)
        src_paths = []
        src_hashes = []
    else:
        src_paths = [p for (p, _, _) in source_index]
        src_colors = np.array([c for (_, c, _) in source_index], dtype=np.float32)
        src_hashes = [ah for (_, _, ah) in source_index]

    # Build hash-based clusters (union-find) for source images
    n_src = len(src_hashes)
    cluster_id = list(range(n_src))

    def find(a):
        while cluster_id[a] != a:
            cluster_id[a] = cluster_id[cluster_id[a]]
            a = cluster_id[a]
        return a

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            cluster_id[rb] = ra

    if n_src > 1 and hash_threshold >= 0:
        for i in range(n_src):
            for j in range(i + 1, n_src):
                # small optimization: quick color check before hash (skip expensive hamming if very different avg)
                if np.linalg.norm(src_colors[i] - src_colors[j]) > 60:  # very different color, skip
                    continue
                if hamming_distance(src_hashes[i], src_hashes[j]) <= hash_threshold:
                    union(i, j)
    # compress cluster ids
    for i in range(n_src):
        cluster_id[i] = find(i)
    # map cluster root to small consecutive ids
    cluster_map = {}
    next_cluster = 0
    for i in range(n_src):
        r = cluster_id[i]
        if r not in cluster_map:
            cluster_map[r] = next_cluster
            next_cluster += 1
    src_cluster = [cluster_map[cluster_id[i]] if n_src > 0 else -1 for i in range(n_src)]

    # First pass: find best-match index for every tile
    n_tiles = len(tiles)
    match_idx = [-1] * n_tiles
    tile_avgs = [None] * n_tiles
    for i, (x, y, tw, th) in enumerate(tiles):
        if stop_flag and stop_flag():
            raise RuntimeError('Stopped by user')
        region = np_target[y:y+th, x:x+tw]
        if region.size == 0:
            match_idx[i] = -1
            tile_avgs[i] = np.array([0.0, 0.0, 0.0])
            continue
        tile_avg = region.mean(axis=(0, 1))
        tile_avgs[i] = tile_avg
        if src_colors.shape[0] == 0:
            match_idx[i] = -1
        else:
            diffs = src_colors - tile_avg
            dists = np.einsum('ij,ij->i', diffs, diffs)
            match_idx[i] = int(np.argmin(dists))
        if progress_callback and (i % 50 == 0 or i == n_tiles - 1):
            progress_callback(i + 1, n_tiles)

    # If merging disabled, just paste per-tile as before
    mosaic = Image.new('RGB', (w, h))

    if not enable_merge_groups:
        total = n_tiles
        for i, (x, y, tw, th) in enumerate(tiles):
            if stop_flag and stop_flag():
                raise RuntimeError('Stopped by user')
            tile_avg = tile_avgs[i] if tile_avgs[i] is not None else np.array([0.0, 0.0, 0.0])
            if match_idx[i] == -1:
                tile_img = Image.new('RGB', (tw, th), tuple([int(max(0, min(255, c))) for c in tile_avg]))
            else:
                best_path = src_paths[match_idx[i]]
                tile_img = _render_source_for_region(best_path, tw, th, tile_avg, stretch_mode, max_stretch_factor, force_full_stretch)
            mosaic.paste(tile_img, (x, y))
            if progress_callback and (i % 10 == 0 or i == total - 1):
                progress_callback(i + 1, total)
        return mosaic

    # Build adjacency graph for tiles that are close (within gap_tolerance) and belong to the same cluster
    def rects_connect(a, b, gap):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh
        # expanded rects by gap
        if (ax2 + gap < bx) or (bx2 + gap < ax) or (ay2 + gap < by) or (by2 + gap < ay):
            # too far apart even with gap
            return False
        if not allow_diagonal:
            # require that rectangles share an edge or overlap (not just diagonal touching)
            horiz = (ax2 + gap >= bx and bx2 + gap >= ax) and (max(ay, by) < min(ay2, by2))
            vert = (ay2 + gap >= by and by2 + gap >= ay) and (max(ax, bx) < min(ax2, bx2))
            return horiz or vert
        else:
            # diagonal allowed: any overlap or within gap in either axis
            return True

    # collect groups by BFS but keyed by cluster id (cluster could be -1)
    processed = [False] * n_tiles
    groups = []

    for i in range(n_tiles):
        if processed[i]:
            continue
        processed[i] = True
        comp = [i]
        queue = [i]
        while queue:
            u = queue.pop()
            for v in range(n_tiles):
                if processed[v]:
                    continue
                # require match_idx both valid or both -1
                if match_idx[u] == -1 or match_idx[v] == -1:
                    continue
                # same cluster?
                if src_cluster[match_idx[u]] != src_cluster[match_idx[v]]:
                    continue
                if rects_connect(tiles[u], tiles[v], gap_tolerance):
                    processed[v] = True
                    queue.append(v)
                    comp.append(v)
        groups.append(comp)

    # Evaluate groups and either render merged or individually
    groups_info = []  # for preview
    total_groups = len(groups)
    for gi, group in enumerate(groups):
        if stop_flag and stop_flag():
            raise RuntimeError('Stopped by user')
        if len(group) == 0:
            if progress_callback:
                progress_callback(gi + 1, total_groups)
            continue
        # compute bbox and coverage
        xs = [tiles[i][0] for i in group]
        ys = [tiles[i][1] for i in group]
        rights = [tiles[i][0] + tiles[i][2] for i in group]
        bottoms = [tiles[i][1] + tiles[i][3] for i in group]
        min_x = min(xs)
        min_y = min(ys)
        max_r = max(rights)
        max_b = max(bottoms)
        bbox_w = max_r - min_x
        bbox_h = max_b - min_y
        area_sum = sum(tiles[i][2] * tiles[i][3] for i in group)
        bbox_area = max(1, bbox_w * bbox_h)
        coverage = area_sum / bbox_area

        # group stats for preview
        groups_info.append((min_x, min_y, bbox_w, bbox_h, coverage, src_cluster[match_idx[group[0]]] if match_idx[group[0]] != -1 else -1, len(group)))

        # decide whether to merge
        can_merge = (coverage >= merge_area_threshold and len(group) >= min_group_tiles and bbox_w * bbox_h >= min_merge_bbox_area and match_idx[group[0]] != -1)

        if preview_only:
            continue

        if can_merge:
            src_index_used = match_idx[group[0]]
            best_path = src_paths[src_index_used]
            avg_combined = np.mean([tile_avgs[i] for i in group], axis=0)
            tile_img = _render_source_for_region(best_path, bbox_w, bbox_h, avg_combined, stretch_mode, max_stretch_factor, force_full_stretch)
            mosaic.paste(tile_img, (min_x, min_y))
        else:
            for i in group:
                x, y, tw, th = tiles[i]
                tile_avg = tile_avgs[i] if tile_avgs[i] is not None else np.array([0.0, 0.0, 0.0])
                if match_idx[i] == -1:
                    tile_img = Image.new('RGB', (tw, th), tuple([int(max(0, min(255, c))) for c in tile_avg]))
                else:
                    best_path = src_paths[match_idx[i]]
                    tile_img = _render_source_for_region(best_path, tw, th, tile_avg, stretch_mode, max_stretch_factor, force_full_stretch)
                mosaic.paste(tile_img, (x, y))

        if progress_callback and (gi % 10 == 0 or gi == total_groups - 1):
            progress_callback(gi + 1, total_groups)

    if preview_only:
        # when preview_only, return groups_info so GUI can overlay
        return None, groups_info
    return mosaic


# ---------- GUI ----------

class MosaicApp:
    def __init__(self, root):
        self.root = root
        root.title('Adaptive Photo Mosaic')
        self.src_index = []
        self.stop = False

        frm = ttk.Frame(root, padding=8)
        frm.grid(row=0, column=0, sticky='nsew')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # --- file selectors ---
        r = 0
        ttk.Label(frm, text='Target image:').grid(row=r, column=0, sticky='w')
        self.target_entry = ttk.Entry(frm, width=50)
        self.target_entry.grid(row=r, column=1, sticky='ew')
        ttk.Button(frm, text='Browse', command=self.browse_target).grid(row=r, column=2)
        r += 1

        ttk.Label(frm, text='Source folder:').grid(row=r, column=0, sticky='w')
        self.src_entry = ttk.Entry(frm, width=50)
        self.src_entry.grid(row=r, column=1, sticky='ew')
        ttk.Button(frm, text='Browse', command=self.browse_src).grid(row=r, column=2)
        r += 1

        ttk.Label(frm, text='Output folder:').grid(row=r, column=0, sticky='w')
        self.out_entry = ttk.Entry(frm, width=50)
        self.out_entry.grid(row=r, column=1, sticky='ew')
        ttk.Button(frm, text='Browse', command=self.browse_out).grid(row=r, column=2)
        r += 1

        # --- options ---
        self.half_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='Half original / half mosaic', variable=self.half_var).grid(row=r, column=0, columnspan=2, sticky='w')
        r += 1

        ttk.Label(frm, text='Split position (% from left):').grid(row=r, column=0, sticky='w')
        self.split_scale = ttk.Scale(frm, from_=10, to=90, orient='horizontal')
        self.split_scale.set(50)
        self.split_scale.grid(row=r, column=1, sticky='ew')
        r += 1

        ttk.Label(frm, text='Min tile size (px):').grid(row=r, column=0, sticky='w')
        self.min_tile = tk.IntVar(value=32)
        ttk.Spinbox(frm, from_=4, to=256, textvariable=self.min_tile).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Max tile size (px):').grid(row=r, column=0, sticky='w')
        self.max_tile = tk.IntVar(value=256)
        ttk.Spinbox(frm, from_=32, to=1024, textvariable=self.max_tile).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Variance threshold:').grid(row=r, column=0, sticky='w')
        self.var_threshold = tk.IntVar(value=650)
        ttk.Spinbox(frm, from_=10, to=5000, textvariable=self.var_threshold).grid(row=r, column=1, sticky='w')
        r += 1

        # NEW: opacity blending
        ttk.Label(frm, text='Mosaic opacity (0 = mosaic only, 100 = original only):').grid(row=r, column=0, sticky='w')
        self.opacity_scale = ttk.Scale(frm, from_=0, to=100, orient='horizontal')
        self.opacity_scale.set(0)
        self.opacity_scale.grid(row=r, column=1, sticky='ew')
        r += 1

        # NEW: stretch / scaling mode
        ttk.Label(frm, text='Tile fit mode:').grid(row=r, column=0, sticky='w')
        self.stretch_mode = tk.StringVar(value='Keep aspect (crop)')
        self.stretch_combo = ttk.Combobox(frm, textvariable=self.stretch_mode, state='readonly', width=30)
        self.stretch_combo['values'] = ('Keep aspect (crop)', 'Scale to fit (letterbox)', 'Stretch (per-axis)')
        self.stretch_combo.grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Max stretch (% per axis, 100 = no stretch):').grid(row=r, column=0, sticky='w')
        self.max_stretch = tk.IntVar(value=300)
        ttk.Spinbox(frm, from_=100, to=1000, textvariable=self.max_stretch).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Force full stretch (ignore limit):').grid(row=r, column=0, sticky='w')
        self.force_stretch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text='Force full stretch', variable=self.force_stretch_var).grid(row=r, column=1, sticky='w')
        r += 1

        # NEW: merge adjacent same-source tiles
        ttk.Label(frm, text='Merge adjacent identical tiles:').grid(row=r, column=0, sticky='w')
        self.merge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='Enable merge groups', variable=self.merge_var).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Merge coverage threshold (% of bounding box):').grid(row=r, column=0, sticky='w')
        self.merge_thresh = tk.IntVar(value=85)
        ttk.Spinbox(frm, from_=10, to=100, textvariable=self.merge_thresh).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Hash hamming threshold (cluster similar sources):').grid(row=r, column=0, sticky='w')
        self.hash_thresh = tk.IntVar(value=6)
        ttk.Spinbox(frm, from_=0, to=64, textvariable=self.hash_thresh).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Allow diagonal adjacency:').grid(row=r, column=0, sticky='w')
        self.diag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='Allow diagonal', variable=self.diag_var).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Gap tolerance (px) for adjacency:').grid(row=r, column=0, sticky='w')
        self.gap_tol = tk.IntVar(value=2)
        ttk.Spinbox(frm, from_=0, to=50, textvariable=self.gap_tol).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Min tiles for merge:').grid(row=r, column=0, sticky='w')
        self.min_tiles_merge = tk.IntVar(value=3)
        ttk.Spinbox(frm, from_=1, to=100, textvariable=self.min_tiles_merge).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Min bbox area (px) for merge:').grid(row=r, column=0, sticky='w')
        self.min_area_merge = tk.IntVar(value=5000)
        ttk.Spinbox(frm, from_=0, to=1000000, textvariable=self.min_area_merge).grid(row=r, column=1, sticky='w')
        r += 1

        # --- actions ---
        buttons = ttk.Frame(frm)
        buttons.grid(row=r, column=0, columnspan=3, pady=(6, 6), sticky='ew')
        ttk.Button(buttons, text='Scan source folder (build color index)', command=self.start_scan_sources).pack(side='left')
        ttk.Button(buttons, text='Preview merge groups', command=self.start_preview_groups).pack(side='left', padx=(6, 0))
        ttk.Button(buttons, text='Generate Mosaic', command=self.start_generate).pack(side='left', padx=(6, 0))
        ttk.Button(buttons, text='Stop', command=self.request_stop).pack(side='left', padx=(6, 0))
        r += 1

        # progress and log
        self.progress = ttk.Progressbar(frm, mode='determinate')
        self.progress.grid(row=r, column=0, columnspan=3, sticky='ew')
        r += 1
        self.status = ttk.Label(frm, text='Ready')
        self.status.grid(row=r, column=0, columnspan=3, sticky='w')
        r += 1

        # preview area
        self.preview_label = ttk.Label(frm, text='Preview will appear here after generation')
        self.preview_label.grid(row=r, column=0, columnspan=3, sticky='nsew')
        frm.rowconfigure(r, weight=1)
        self.preview_imgtk = None

    # ---------- GUI callbacks ----------
    def browse_target(self):
        p = filedialog.askopenfilename(filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp;*.tiff')])
        if p:
            self.target_entry.delete(0, tk.END)
            self.target_entry.insert(0, p)

    def browse_src(self):
        p = filedialog.askdirectory()
        if p:
            self.src_entry.delete(0, tk.END)
            self.src_entry.insert(0, p)

    def browse_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, p)

    def set_status(self, txt):
        def _():
            self.status.config(text=txt)
        self.root.after(0, _)

    def update_progress(self, v, total):
        def _():
            self.progress['maximum'] = total
            self.progress['value'] = v
            self.status.config(text=f'Progress: {v}/{total}')
        self.root.after(0, _)

    def start_scan_sources(self):
        folder = self.src_entry.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror('Error', 'Select a valid source folder first')
            return
        t = threading.Thread(target=self.scan_sources_thread, args=(folder,), daemon=True)
        t.start()

    def scan_sources_thread(self, folder):
        self.set_status('Scanning source images...')
        def prog(i, n):
            self.update_progress(i, n)
        try:
            self.src_index = build_source_index(folder, thumb_for_avg=32, hash_size=8, progress_callback=prog)
            self.set_status(f'Scan complete: {len(self.src_index)} images')
        except Exception as e:
            self.set_status('Error scanning sources')
            messagebox.showerror('Error', str(e))

    def request_stop(self):
        self.stop = True
        self.set_status('Stop requested...')

    def start_preview_groups(self):
        tpath = self.target_entry.get().strip()
        sfolder = self.src_entry.get().strip()
        if not tpath or not os.path.isfile(tpath):
            messagebox.showerror('Error', 'Select a valid target image.')
            return
        if not sfolder or not os.path.isdir(sfolder):
            messagebox.showerror('Error', 'Select a valid source folder and build index (or press scan).')
            return
        if not self.src_index:
            self.set_status('Index empty - scanning now...')
            try:
                self.src_index = build_source_index(sfolder, thumb_for_avg=32, hash_size=8, progress_callback=self.update_progress)
            except Exception as e:
                messagebox.showerror('Error', f'Failed to scan source folder: {e}')
                return
        thread = threading.Thread(target=self.preview_groups_thread, daemon=True)
        thread.start()

    def preview_groups_thread(self):
        try:
            self.set_status('Loading target image for preview...')
            with Image.open(self.target_entry.get().strip()) as targ:
                targ = targ.convert('RGB')
                max_dim = 1000
                if max(targ.size) > max_dim:
                    if targ.size[0] >= targ.size[1]:
                        targ = targ.resize((max_dim, int(targ.size[1] * max_dim / targ.size[0])), Image.Resampling.LANCZOS)
                    else:
                        targ = targ.resize((int(targ.size[0] * max_dim / targ.size[1]), max_dim), Image.Resampling.LANCZOS)
                np_t = np.asarray(targ, dtype=np.float32)

                tiles = quadtree_tiles(np_t, min_tile=int(self.min_tile.get()), max_tile=int(self.max_tile.get()), var_threshold=float(self.var_threshold.get()))

                # call create_mosaic_from_tiles in preview_only mode to get groups_info
                _, groups_info = create_mosaic_from_tiles(targ, tiles, self.src_index, progress_callback=self.update_progress, stop_flag=lambda: self.stop,
                                                          stretch_mode='keep', max_stretch_factor=float(self.max_stretch.get())/100.0, force_full_stretch=bool(self.force_stretch_var.get()),
                                                          enable_merge_groups=bool(self.merge_var.get()), merge_area_threshold=float(self.merge_thresh.get())/100.0,
                                                          hash_threshold=int(self.hash_thresh.get()), allow_diagonal=bool(self.diag_var.get()), gap_tolerance=int(self.gap_tol.get()),
                                                          min_group_tiles=int(self.min_tiles_merge.get()), min_merge_bbox_area=int(self.min_area_merge.get()), preview_only=True)

                # draw overlay boxes for groups with high coverage
                overlay = Image.new('RGBA', targ.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                for (mx, my, mw, mh, cov, cid, nt) in groups_info:
                    if cov >= float(self.merge_thresh.get())/100.0 and nt >= int(self.min_tiles_merge.get()) and (mw*mh) >= int(self.min_area_merge.get()):
                        color = tuple([random.randint(32, 220) for _ in range(3)])
                        draw.rectangle([mx, my, mx + mw - 1, my + mh - 1], outline=color + (255,), width=3)
                        draw.rectangle([mx, my, mx + mw - 1, my + mh - 1], fill=color + (50,))

                combined = targ.convert('RGBA')
                combined = Image.alpha_composite(combined, overlay)
                combined.thumbnail((800, 600), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(combined)

                def show():
                    self.preview_label.config(image=imgtk, text='')
                    self.preview_imgtk = imgtk
                    self.set_status('Preview showing merged groups (colored). Generate to apply.')
                self.root.after(0, show)
        except Exception as e:
            self.set_status('Error while previewing')
            messagebox.showerror('Error', str(e))

    def start_generate(self):
        tpath = self.target_entry.get().strip()
        sfolder = self.src_entry.get().strip()
        outfolder = self.out_entry.get().strip()
        if not tpath or not os.path.isfile(tpath):
            messagebox.showerror('Error', 'Select a valid target image.')
            return
        if not sfolder or not os.path.isdir(sfolder):
            messagebox.showerror('Error', 'Select a valid source folder and build index (or press scan).')
            return
        if not outfolder or not os.path.isdir(outfolder):
            messagebox.showerror('Error', 'Select a valid output folder.')
            return

        # ensure we have a cache - if user hasn't scanned, build quickly
        if not self.src_index:
            self.set_status('Index empty - scanning now...')
            try:
                self.src_index = build_source_index(sfolder, thumb_for_avg=32, hash_size=8, progress_callback=self.update_progress)
            except Exception as e:
                messagebox.showerror('Error', f'Failed to scan source folder: {e}')
                return

        # read parameters
        min_t = int(self.min_tile.get())
        max_t = int(self.max_tile.get())
        var_th = float(self.var_threshold.get())
        half = bool(self.half_var.get())
        split_pct = float(self.split_scale.get()) / 100.0
        opacity = float(self.opacity_scale.get()) / 100.0

        # translate stretch mode and max stretch
        sm = self.stretch_combo.get()
        if sm == 'Keep aspect (crop)':
            sm_code = 'keep'
        elif sm == 'Scale to fit (letterbox)':
            sm_code = 'fit'
        else:
            sm_code = 'stretch'
        max_stretch_pct = float(self.max_stretch.get())
        max_stretch_factor = max_stretch_pct / 100.0
        force_full = bool(self.force_stretch_var.get())

        # merge options
        merge_enabled = bool(self.merge_var.get())
        merge_thresh = float(self.merge_thresh.get()) / 100.0
        hash_thresh = int(self.hash_thresh.get())
        allow_diag = bool(self.diag_var.get())
        gap_tol = int(self.gap_tol.get())
        min_tiles_merge = int(self.min_tiles_merge.get())
        min_area_merge = int(self.min_area_merge.get())

        thread = threading.Thread(target=self.generate_thread, args=(tpath, outfolder, min_t, max_t, var_th, half, split_pct, opacity, sm_code, max_stretch_factor, force_full, merge_enabled, merge_thresh, hash_thresh, allow_diag, gap_tol, min_tiles_merge, min_area_merge), daemon=True)
        self.stop = False
        thread.start()

    def generate_thread(self, target_path, outfolder, min_t, max_t, var_th, half, split_pct, opacity, stretch_mode, max_stretch_factor, force_full, merge_enabled, merge_thresh, hash_thresh, allow_diag, gap_tol, min_tiles_merge, min_area_merge):
        try:
            self.set_status('Loading target image...')
            with Image.open(target_path) as targ:
                targ = targ.convert('RGB')
                # optionally downscale target if very large for reasonable runtime
                max_dim = 1200
                if max(targ.size) > max_dim:
                    # keep aspect while limiting max dimension
                    if targ.size[0] >= targ.size[1]:
                        targ = targ.resize((max_dim, int(targ.size[1] * max_dim / targ.size[0])), Image.Resampling.LANCZOS)
                    else:
                        targ = targ.resize((int(targ.size[0] * max_dim / targ.size[1]), max_dim), Image.Resampling.LANCZOS)

                t_w, t_h = targ.size
                self.set_status('Preparing image array...')
                np_t = np.asarray(targ, dtype=np.float32)

                self.set_status('Computing tiles (quadtree)...')
                tiles = quadtree_tiles(np_t, min_tile=min_t, max_tile=max_t, var_threshold=var_th)
                self.set_status(f'{len(tiles)} tiles computed')

                def prog(i, n):
                    self.update_progress(i, n)

                # wrapper for stop flag
                stop_flag = lambda : self.stop

                self.set_status('Building mosaic (this may take a while)')
                mosaic = create_mosaic_from_tiles(targ, tiles, self.src_index, progress_callback=prog, stop_flag=stop_flag,
                                                 stretch_mode=stretch_mode, max_stretch_factor=max_stretch_factor, force_full_stretch=force_full,
                                                 enable_merge_groups=merge_enabled, merge_area_threshold=merge_thresh,
                                                 hash_threshold=hash_thresh, allow_diagonal=allow_diag, gap_tolerance=gap_tol,
                                                 min_group_tiles=min_tiles_merge, min_merge_bbox_area=min_area_merge)

                # apply opacity blending: if half mode, blend only the mosaic side with original
                if half:
                    split_x = int(t_w * split_pct)
                    out = Image.new('RGB', (t_w, t_h))
                    out.paste(targ.crop((0, 0, split_x, t_h)), (0, 0))

                    mosaic_right = mosaic.crop((split_x, 0, t_w, t_h))
                    orig_right = targ.crop((split_x, 0, t_w, t_h))
                    if opacity > 0:
                        blended_right = Image.blend(mosaic_right, orig_right, opacity)
                    else:
                        blended_right = mosaic_right
                    out.paste(blended_right, (split_x, 0))
                else:
                    if opacity > 0:
                        out = Image.blend(mosaic, targ, opacity)
                    else:
                        out = mosaic

                timestamp = int(time.time())
                outpath = os.path.join(outfolder, f'mosaic_{timestamp}.jpg')
                out.save(outpath, quality=90)
                self.set_status(f'Done. Saved to {outpath}')

                # update preview (scale down for display)
                preview = out.copy()
                preview.thumbnail((800, 600), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(preview)

                def show_preview():
                    self.preview_label.config(image=imgtk, text='')
                    self.preview_imgtk = imgtk
                self.root.after(0, show_preview)

        except Exception as ex:
            if str(ex) == 'Stopped by user':
                self.set_status('Generation stopped.')
            else:
                self.set_status('Error while generating')
                messagebox.showerror('Error', str(ex))


# ---------- Main ----------

if __name__ == '__main__':
    root = tk.Tk()
    app = MosaicApp(root)
    root.geometry('1000x980')
    root.mainloop()