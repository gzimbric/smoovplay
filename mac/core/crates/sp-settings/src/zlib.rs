//! Raw deflate as zlib writes it at level 9 (zlib.compressobj(9, zlib.DEFLATED, -15)), inflate, and crc32, for share
//! codes. The compressor is a line-by-line port of zlib's deflate_slow, longest_match and trees.c (1.2.12 to 1.3.x
//! give the same output for inputs this small), so a look's code is the same here as from the Python, bit for bit.
//! Inputs must fit in one window (32 KB). A share code's body is under 3 KB.

// Indexed loops, as in zlib's C, so the two read side by side.
#![allow(clippy::needless_range_loop)]

// ------------------------------------------------------------------------------------------------ crc32
fn crc_table() -> &'static [u32; 256] {
    static T: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; 256];
        for (n, e) in t.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        t
    })
}

/// zlib.crc32(data).
pub fn crc32(data: &[u8]) -> u32 {
    let t = crc_table();
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = t[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

// ------------------------------------------------------------------------------------------------ deflate
const MAX_BITS: usize = 15;
const MAX_BL_BITS: usize = 7;
const LENGTH_CODES: usize = 29;
const LITERALS: usize = 256;
const L_CODES: usize = LITERALS + 1 + LENGTH_CODES;
const D_CODES: usize = 30;
const BL_CODES: usize = 19;
const HEAP_SIZE: usize = 2 * L_CODES + 1;
const END_BLOCK: usize = 256;
const REP_3_6: usize = 16;
const REPZ_3_10: usize = 17;
const REPZ_11_138: usize = 18;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const W_BITS: usize = 15;
const W_SIZE: usize = 1 << W_BITS;
const W_MASK: usize = W_SIZE - 1;
const HASH_BITS: usize = 8 + 7; // memLevel 8
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: usize = HASH_SIZE - 1;
const HASH_SHIFT: usize = HASH_BITS.div_ceil(MIN_MATCH);
const MIN_LOOKAHEAD: usize = MAX_MATCH + MIN_MATCH + 1;
const MAX_DIST: usize = W_SIZE - MIN_LOOKAHEAD;
const TOO_FAR: usize = 4096;
// level 9: good, lazy, nice, chain
const GOOD_MATCH: usize = 32;
const MAX_LAZY: usize = 258;
const NICE_MATCH: usize = 258;
const MAX_CHAIN: usize = 4096;

const EXTRA_LBITS: [u8; LENGTH_CODES] =
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const EXTRA_DBITS: [u8; D_CODES] =
    [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const EXTRA_BLBITS: [u8; BL_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];
const BL_ORDER: [usize; BL_CODES] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// A tree element. fc is Freq or Code and dl is Dad or Len, as unions in C, and trees.c relies on the sharing.
#[derive(Clone, Copy, Default)]
struct Ct {
    fc: u16,
    dl: u16,
}

struct Static {
    ltree: [Ct; L_CODES + 2],
    dtree: [Ct; D_CODES],
    dist_code: [u8; 512],
    length_code: [u8; 256],
    base_length: [i32; LENGTH_CODES],
    base_dist: [i32; D_CODES],
}

fn bi_reverse(mut code: u32, mut len: u32) -> u32 {
    let mut res = 0;
    loop {
        res |= code & 1;
        code >>= 1;
        res <<= 1;
        len -= 1;
        if len == 0 {
            break;
        }
    }
    res >> 1
}

fn gen_codes(tree: &mut [Ct], max_code: isize, bl_count: &[u16; MAX_BITS + 1]) {
    let mut next_code = [0u16; MAX_BITS + 1];
    let mut code: u32 = 0;
    for bits in 1..=MAX_BITS {
        code = (code + bl_count[bits - 1] as u32) << 1;
        next_code[bits] = code as u16;
    }
    for n in 0..=max_code.max(-1) {
        if n < 0 {
            break;
        }
        let len = tree[n as usize].dl as usize;
        if len == 0 {
            continue;
        }
        tree[n as usize].fc = bi_reverse(next_code[len] as u32, len as u32) as u16;
        next_code[len] = next_code[len].wrapping_add(1);
    }
}

fn statics() -> &'static Static {
    static S: std::sync::OnceLock<Static> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let mut s = Static {
            ltree: [Ct::default(); L_CODES + 2],
            dtree: [Ct::default(); D_CODES],
            dist_code: [0; 512],
            length_code: [0; 256],
            base_length: [0; LENGTH_CODES],
            base_dist: [0; D_CODES],
        };
        let mut length = 0usize;
        let mut code = 0usize;
        while code < LENGTH_CODES - 1 {
            s.base_length[code] = length as i32;
            for _ in 0..(1 << EXTRA_LBITS[code]) {
                s.length_code[length] = code as u8;
                length += 1;
            }
            code += 1;
        }
        s.length_code[length - 1] = code as u8;
        let mut dist = 0usize;
        code = 0;
        while code < 16 {
            s.base_dist[code] = dist as i32;
            for _ in 0..(1 << EXTRA_DBITS[code]) {
                s.dist_code[dist] = code as u8;
                dist += 1;
            }
            code += 1;
        }
        dist >>= 7;
        while code < D_CODES {
            s.base_dist[code] = (dist << 7) as i32;
            for _ in 0..(1 << (EXTRA_DBITS[code] - 7)) {
                s.dist_code[256 + dist] = code as u8;
                dist += 1;
            }
            code += 1;
        }
        let mut bl_count = [0u16; MAX_BITS + 1];
        let mut n = 0;
        while n <= 143 {
            s.ltree[n].dl = 8;
            bl_count[8] += 1;
            n += 1;
        }
        while n <= 255 {
            s.ltree[n].dl = 9;
            bl_count[9] += 1;
            n += 1;
        }
        while n <= 279 {
            s.ltree[n].dl = 7;
            bl_count[7] += 1;
            n += 1;
        }
        while n <= 287 {
            s.ltree[n].dl = 8;
            bl_count[8] += 1;
            n += 1;
        }
        gen_codes(&mut s.ltree, (L_CODES + 1) as isize, &bl_count);
        for n in 0..D_CODES {
            s.dtree[n].dl = 5;
            s.dtree[n].fc = bi_reverse(n as u32, 5) as u16;
        }
        s
    })
}

#[derive(Clone, Copy, PartialEq)]
enum TreeKind {
    Lit,
    Dist,
    Bl,
}

struct Deflate {
    window: Vec<u8>,
    prev: Vec<u16>,
    head: Vec<u16>,
    ins_h: usize,
    strstart: usize,
    lookahead: usize,
    block_start: isize,
    match_length: usize,
    prev_length: usize,
    prev_match: usize,
    match_start: usize,
    match_available: bool,
    // trees
    dyn_ltree: [Ct; HEAP_SIZE],
    dyn_dtree: [Ct; 2 * D_CODES + 1],
    bl_tree: [Ct; 2 * BL_CODES + 1],
    l_max_code: isize,
    d_max_code: isize,
    bl_max_code: isize,
    bl_count: [u16; MAX_BITS + 1],
    heap: [usize; 2 * L_CODES + 1],
    heap_len: usize,
    heap_max: usize,
    depth: [u8; 2 * L_CODES + 1],
    syms: Vec<(u16, u8)>, // (dist, lc)
    opt_len: u64,
    static_len: u64,
    bi_buf: u16,
    bi_valid: i32,
    out: Vec<u8>,
}

impl Deflate {
    fn tree(&mut self, k: TreeKind) -> &mut [Ct] {
        match k {
            TreeKind::Lit => &mut self.dyn_ltree,
            TreeKind::Dist => &mut self.dyn_dtree,
            TreeKind::Bl => &mut self.bl_tree,
        }
    }

    fn put_short(&mut self, w: u16) {
        self.out.push((w & 0xff) as u8);
        self.out.push((w >> 8) as u8);
    }

    fn send_bits(&mut self, value: i32, length: i32) {
        // In C the shifts happen in int, then the result is cut to 16 bits.
        let val = value as u16 as u32;
        if self.bi_valid > 16 - length {
            self.bi_buf |= (val << self.bi_valid) as u16;
            let b = self.bi_buf;
            self.put_short(b);
            self.bi_buf = (val >> (16 - self.bi_valid)) as u16;
            self.bi_valid += length - 16;
        } else {
            self.bi_buf |= (val << self.bi_valid) as u16;
            self.bi_valid += length;
        }
    }

    fn send_code(&mut self, c: usize, tree: &[Ct]) {
        self.send_bits(tree[c].fc as i32, tree[c].dl as i32);
    }

    fn bi_windup(&mut self) {
        if self.bi_valid > 8 {
            let b = self.bi_buf;
            self.put_short(b);
        } else if self.bi_valid > 0 {
            self.out.push(self.bi_buf as u8);
        }
        self.bi_buf = 0;
        self.bi_valid = 0;
    }

    fn init_block(&mut self) {
        for n in 0..L_CODES {
            self.dyn_ltree[n].fc = 0;
        }
        for n in 0..D_CODES {
            self.dyn_dtree[n].fc = 0;
        }
        for n in 0..BL_CODES {
            self.bl_tree[n].fc = 0;
        }
        self.dyn_ltree[END_BLOCK].fc = 1;
        self.opt_len = 0;
        self.static_len = 0;
        self.syms.clear();
    }

    fn d_code(&self, dist: usize) -> usize {
        let st = statics();
        if dist < 256 { st.dist_code[dist] as usize } else { st.dist_code[256 + (dist >> 7)] as usize }
    }

    fn tally_lit(&mut self, c: u8) {
        self.syms.push((0, c));
        self.dyn_ltree[c as usize].fc += 1;
    }

    fn tally_dist(&mut self, dist: usize, lc: usize) {
        self.syms.push((dist as u16, lc as u8));
        let dist = dist - 1;
        let lcode = statics().length_code[lc] as usize;
        self.dyn_ltree[lcode + LITERALS + 1].fc += 1;
        let dc = self.d_code(dist);
        self.dyn_dtree[dc].fc += 1;
    }

    fn smaller(tree: &[Ct], depth: &[u8], n: usize, m: usize) -> bool {
        tree[n].fc < tree[m].fc || (tree[n].fc == tree[m].fc && depth[n] <= depth[m])
    }

    fn pqdownheap(&mut self, k_tree: TreeKind, mut k: usize) {
        let v = self.heap[k];
        let mut j = k << 1;
        while j <= self.heap_len {
            {
                let (heap, depth) = (&self.heap, &self.depth);
                let tree: &[Ct] = match k_tree {
                    TreeKind::Lit => &self.dyn_ltree,
                    TreeKind::Dist => &self.dyn_dtree,
                    TreeKind::Bl => &self.bl_tree,
                };
                if j < self.heap_len && Self::smaller(tree, depth, heap[j + 1], heap[j]) {
                    j += 1;
                }
                if Self::smaller(tree, depth, v, heap[j]) {
                    break;
                }
            }
            self.heap[k] = self.heap[j];
            k = j;
            j <<= 1;
        }
        self.heap[k] = v;
    }

    fn gen_bitlen(&mut self, kind: TreeKind) {
        let st = statics();
        let (max_code, stree, extra, base, max_length): (isize, Option<&[Ct]>, &[u8], usize, usize) = match kind {
            TreeKind::Lit => (self.l_max_code, Some(&st.ltree[..]), &EXTRA_LBITS[..], LITERALS + 1, MAX_BITS),
            TreeKind::Dist => (self.d_max_code, Some(&st.dtree[..]), &EXTRA_DBITS[..], 0, MAX_BITS),
            TreeKind::Bl => (self.bl_max_code, None, &EXTRA_BLBITS[..], 0, MAX_BL_BITS),
        };
        let mut overflow = 0i32;
        self.bl_count = [0; MAX_BITS + 1];
        let root = self.heap[self.heap_max];
        self.tree(kind)[root].dl = 0;
        let mut h = self.heap_max + 1;
        while h < HEAP_SIZE {
            let n = self.heap[h];
            let tree = self.tree(kind);
            let mut bits = tree[tree[n].dl as usize].dl as usize + 1;
            if bits > max_length {
                bits = max_length;
                overflow += 1;
            }
            tree[n].dl = bits as u16;
            h += 1;
            if n as isize > max_code {
                continue;
            }
            let f = tree[n].fc as u64;
            self.bl_count[bits] += 1;
            let xbits = if n >= base { extra[n - base] as u64 } else { 0 };
            self.opt_len = self.opt_len.wrapping_add(f * (bits as u64 + xbits));
            if let Some(stree) = stree {
                self.static_len = self.static_len.wrapping_add(f * (stree[n].dl as u64 + xbits));
            }
        }
        if overflow == 0 {
            return;
        }
        loop {
            let mut bits = max_length - 1;
            while self.bl_count[bits] == 0 {
                bits -= 1;
            }
            self.bl_count[bits] -= 1;
            self.bl_count[bits + 1] += 2;
            self.bl_count[max_length] -= 1;
            overflow -= 2;
            if overflow <= 0 {
                break;
            }
        }
        let mut h = HEAP_SIZE;
        let mut bits = max_length;
        while bits != 0 {
            let mut n = self.bl_count[bits] as usize;
            while n != 0 {
                h -= 1;
                let m = self.heap[h];
                if m as isize > max_code {
                    continue;
                }
                let tree = self.tree(kind);
                if tree[m].dl as usize != bits {
                    let (len, freq) = (tree[m].dl as u64, tree[m].fc as u64);
                    tree[m].dl = bits as u16;
                    self.opt_len = self.opt_len.wrapping_add((bits as u64).wrapping_sub(len).wrapping_mul(freq));
                }
                n -= 1;
            }
            bits -= 1;
        }
    }

    fn build_tree(&mut self, kind: TreeKind) {
        let st = statics();
        let (stree, elems): (Option<&[Ct]>, usize) = match kind {
            TreeKind::Lit => (Some(&st.ltree[..]), L_CODES),
            TreeKind::Dist => (Some(&st.dtree[..]), D_CODES),
            TreeKind::Bl => (None, BL_CODES),
        };
        let mut max_code: isize = -1;
        self.heap_len = 0;
        self.heap_max = HEAP_SIZE;
        for n in 0..elems {
            if self.tree(kind)[n].fc != 0 {
                self.heap_len += 1;
                self.heap[self.heap_len] = n;
                max_code = n as isize;
                self.depth[n] = 0;
            } else {
                self.tree(kind)[n].dl = 0;
            }
        }
        while self.heap_len < 2 {
            let node = if max_code < 2 {
                max_code += 1;
                max_code as usize
            } else {
                0
            };
            self.heap_len += 1;
            self.heap[self.heap_len] = node;
            self.tree(kind)[node].fc = 1;
            self.depth[node] = 0;
            self.opt_len = self.opt_len.wrapping_sub(1);
            if let Some(stree) = stree {
                self.static_len = self.static_len.wrapping_sub(stree[node].dl as u64);
            }
        }
        match kind {
            TreeKind::Lit => self.l_max_code = max_code,
            TreeKind::Dist => self.d_max_code = max_code,
            TreeKind::Bl => self.bl_max_code = max_code,
        }
        let mut n = self.heap_len / 2;
        while n >= 1 {
            self.pqdownheap(kind, n);
            n -= 1;
        }
        let mut node = elems;
        loop {
            // pqremove
            let n = self.heap[1];
            self.heap[1] = self.heap[self.heap_len];
            self.heap_len -= 1;
            self.pqdownheap(kind, 1);
            let m = self.heap[1];
            self.heap_max -= 1;
            self.heap[self.heap_max] = n;
            self.heap_max -= 1;
            self.heap[self.heap_max] = m;
            let d = self.depth[n].max(self.depth[m]).wrapping_add(1);
            let tree = self.tree(kind);
            tree[node].fc = tree[n].fc.wrapping_add(tree[m].fc);
            tree[n].dl = node as u16;
            tree[m].dl = node as u16;
            self.depth[node] = d;
            self.heap[1] = node;
            node += 1;
            self.pqdownheap(kind, 1);
            if self.heap_len < 2 {
                break;
            }
        }
        self.heap_max -= 1;
        self.heap[self.heap_max] = self.heap[1];
        self.gen_bitlen(kind);
        let bl_count = self.bl_count;
        gen_codes(self.tree(kind), max_code, &bl_count);
    }

    fn scan_tree(&mut self, kind: TreeKind, max_code: isize) {
        let mut prevlen: i32 = -1;
        let mut nextlen = self.tree(kind)[0].dl as i32;
        let mut count = 0;
        let (mut max_count, mut min_count) = (7, 4);
        if nextlen == 0 {
            max_count = 138;
            min_count = 3;
        }
        self.tree(kind)[(max_code + 1) as usize].dl = 0xffff;
        for n in 0..=max_code as usize {
            let curlen = nextlen;
            nextlen = self.tree(kind)[n + 1].dl as i32;
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                self.bl_tree[curlen as usize].fc += count as u16;
            } else if curlen != 0 {
                if curlen != prevlen {
                    self.bl_tree[curlen as usize].fc += 1;
                }
                self.bl_tree[REP_3_6].fc += 1;
            } else if count <= 10 {
                self.bl_tree[REPZ_3_10].fc += 1;
            } else {
                self.bl_tree[REPZ_11_138].fc += 1;
            }
            count = 0;
            prevlen = curlen;
            if nextlen == 0 {
                max_count = 138;
                min_count = 3;
            } else if curlen == nextlen {
                max_count = 6;
                min_count = 3;
            } else {
                max_count = 7;
                min_count = 4;
            }
        }
    }

    fn send_tree(&mut self, kind: TreeKind, max_code: isize) {
        let mut prevlen: i32 = -1;
        let mut nextlen = self.tree(kind)[0].dl as i32;
        let mut count = 0;
        let (mut max_count, mut min_count) = (7, 4);
        if nextlen == 0 {
            max_count = 138;
            min_count = 3;
        }
        let bl = self.bl_tree;
        for n in 0..=max_code as usize {
            let curlen = nextlen;
            nextlen = self.tree(kind)[n + 1].dl as i32;
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                loop {
                    self.send_code(curlen as usize, &bl);
                    count -= 1;
                    if count == 0 {
                        break;
                    }
                }
            } else if curlen != 0 {
                if curlen != prevlen {
                    self.send_code(curlen as usize, &bl);
                    count -= 1;
                }
                self.send_code(REP_3_6, &bl);
                self.send_bits(count - 3, 2);
            } else if count <= 10 {
                self.send_code(REPZ_3_10, &bl);
                self.send_bits(count - 3, 3);
            } else {
                self.send_code(REPZ_11_138, &bl);
                self.send_bits(count - 11, 7);
            }
            count = 0;
            prevlen = curlen;
            if nextlen == 0 {
                max_count = 138;
                min_count = 3;
            } else if curlen == nextlen {
                max_count = 6;
                min_count = 3;
            } else {
                max_count = 7;
                min_count = 4;
            }
        }
    }

    fn build_bl_tree(&mut self) -> usize {
        let (l, d) = (self.l_max_code, self.d_max_code);
        self.scan_tree(TreeKind::Lit, l);
        self.scan_tree(TreeKind::Dist, d);
        self.build_tree(TreeKind::Bl);
        let mut max_blindex = BL_CODES - 1;
        while max_blindex >= 3 {
            if self.bl_tree[BL_ORDER[max_blindex]].dl != 0 {
                break;
            }
            max_blindex -= 1;
        }
        self.opt_len = self.opt_len.wrapping_add(3 * (max_blindex as u64 + 1) + 5 + 5 + 4);
        max_blindex
    }

    fn send_all_trees(&mut self, lcodes: usize, dcodes: usize, blcodes: usize) {
        self.send_bits(lcodes as i32 - 257, 5);
        self.send_bits(dcodes as i32 - 1, 5);
        self.send_bits(blcodes as i32 - 4, 4);
        for rank in 0..blcodes {
            let len = self.bl_tree[BL_ORDER[rank]].dl as i32;
            self.send_bits(len, 3);
        }
        self.send_tree(TreeKind::Lit, lcodes as isize - 1);
        self.send_tree(TreeKind::Dist, dcodes as isize - 1);
    }

    fn compress_block(&mut self, ltree: &[Ct], dtree: &[Ct]) {
        let st = statics();
        let syms = std::mem::take(&mut self.syms);
        for &(dist, lc) in &syms {
            let (dist, mut lc) = (dist as usize, lc as i32);
            if dist == 0 {
                self.send_code(lc as usize, ltree);
            } else {
                let code = st.length_code[lc as usize] as usize;
                self.send_code(code + LITERALS + 1, ltree);
                let extra = EXTRA_LBITS[code] as i32;
                if extra != 0 {
                    lc -= st.base_length[code];
                    self.send_bits(lc, extra);
                }
                let mut dist = dist - 1;
                let code = self.d_code(dist);
                self.send_code(code, dtree);
                let extra = EXTRA_DBITS[code] as i32;
                if extra != 0 {
                    dist -= st.base_dist[code] as usize;
                    self.send_bits(dist as i32, extra);
                }
            }
        }
        self.syms = syms;
        self.send_code(END_BLOCK, ltree);
    }

    fn stored_block(&mut self, start: usize, stored_len: usize, last: bool) {
        self.send_bits(last as i32, 3); // STORED_BLOCK << 1
        self.bi_windup();
        self.put_short(stored_len as u16);
        self.put_short(!(stored_len as u16));
        let data = self.window[start..start + stored_len].to_vec();
        self.out.extend_from_slice(&data);
    }

    fn flush_block(&mut self, last: bool) {
        let stored_len = (self.strstart as isize - self.block_start) as usize;
        let buf = (self.block_start >= 0).then_some(self.block_start as usize);
        self.build_tree(TreeKind::Lit);
        self.build_tree(TreeKind::Dist);
        let max_blindex = self.build_bl_tree();
        let mut opt_lenb = (self.opt_len.wrapping_add(3 + 7)) >> 3;
        let static_lenb = (self.static_len.wrapping_add(3 + 7)) >> 3;
        if static_lenb <= opt_lenb {
            opt_lenb = static_lenb;
        }
        if let Some(start) = buf.filter(|_| stored_len as u64 + 4 <= opt_lenb) {
            self.stored_block(start, stored_len, last);
        } else if static_lenb == opt_lenb {
            self.send_bits((1 << 1) + last as i32, 3);
            let st = statics();
            self.compress_block(&st.ltree, &st.dtree);
        } else {
            self.send_bits((2 << 1) + last as i32, 3);
            let (l, d) = (self.l_max_code, self.d_max_code);
            self.send_all_trees((l + 1) as usize, (d + 1) as usize, max_blindex + 1);
            let (lt, dt) = (self.dyn_ltree, self.dyn_dtree);
            self.compress_block(&lt, &dt);
        }
        self.init_block();
        if last {
            self.bi_windup();
        }
        self.block_start = self.strstart as isize;
    }

    fn update_hash(&mut self, c: u8) {
        self.ins_h = ((self.ins_h << HASH_SHIFT) ^ c as usize) & HASH_MASK;
    }

    fn insert_string(&mut self, s: usize) -> usize {
        let c = self.window[s + MIN_MATCH - 1];
        self.update_hash(c);
        let head = self.head[self.ins_h];
        self.prev[s & W_MASK] = head;
        self.head[self.ins_h] = s as u16;
        head as usize
    }

    fn longest_match(&mut self, mut cur_match: usize) -> usize {
        let mut chain_length = MAX_CHAIN;
        let scan = self.strstart;
        let mut best_len = self.prev_length;
        let mut nice_match = NICE_MATCH;
        let limit = self.strstart.saturating_sub(MAX_DIST);
        let strend = self.strstart + MAX_MATCH;
        let w = &self.window;
        let mut scan_end1 = w[scan + best_len - 1];
        let mut scan_end = w[scan + best_len];
        if self.prev_length >= GOOD_MATCH {
            chain_length >>= 2;
        }
        if nice_match > self.lookahead {
            nice_match = self.lookahead;
        }
        loop {
            let m = cur_match;
            let skip = w[m + best_len] != scan_end
                || w[m + best_len - 1] != scan_end1
                || w[m] != w[scan]
                || w[m + 1] != w[scan + 1];
            if !skip {
                // scan[2] and match[2] are equal (same hash, HASH_BITS >= 8), so compare from 3, 8 at a time
                let mut s = scan + 2;
                let mut mm = m + 2;
                loop {
                    let mut stop = false;
                    for _ in 0..8 {
                        s += 1;
                        mm += 1;
                        if w[s] != w[mm] {
                            stop = true;
                            break;
                        }
                    }
                    if stop || s >= strend {
                        break;
                    }
                }
                let len = MAX_MATCH - (strend - s);
                if len > best_len {
                    self.match_start = cur_match;
                    best_len = len;
                    if len >= nice_match {
                        break;
                    }
                    scan_end1 = w[scan + best_len - 1];
                    scan_end = w[scan + best_len];
                }
            }
            cur_match = self.prev[cur_match & W_MASK] as usize;
            if cur_match <= limit {
                break;
            }
            chain_length -= 1;
            if chain_length == 0 {
                break;
            }
        }
        best_len.min(self.lookahead)
    }

    fn run(&mut self, input_len: usize) {
        // fill_window takes all of the input at once. The bytes after it read as zeros (WIN_INIT).
        self.lookahead = input_len;
        if self.lookahead >= MIN_MATCH {
            self.ins_h = self.window[0] as usize;
            let c = self.window[1];
            self.update_hash(c);
        }
        loop {
            if self.lookahead == 0 {
                break;
            }
            let mut hash_head = 0;
            if self.lookahead >= MIN_MATCH {
                hash_head = self.insert_string(self.strstart);
            }
            self.prev_length = self.match_length;
            self.prev_match = self.match_start;
            self.match_length = MIN_MATCH - 1;
            if hash_head != 0 && self.prev_length < MAX_LAZY && self.strstart - hash_head <= MAX_DIST {
                self.match_length = self.longest_match(hash_head);
                if self.match_length <= 5
                    && self.match_length == MIN_MATCH
                    && self.strstart - self.match_start > TOO_FAR
                {
                    self.match_length = MIN_MATCH - 1;
                }
            }
            if self.prev_length >= MIN_MATCH && self.match_length <= self.prev_length {
                let max_insert = self.strstart + self.lookahead - MIN_MATCH;
                self.tally_dist(self.strstart - 1 - self.prev_match, self.prev_length - MIN_MATCH);
                self.lookahead -= self.prev_length - 1;
                self.prev_length -= 2;
                loop {
                    self.strstart += 1;
                    if self.strstart <= max_insert {
                        self.insert_string(self.strstart);
                    }
                    self.prev_length -= 1;
                    if self.prev_length == 0 {
                        break;
                    }
                }
                self.match_available = false;
                self.match_length = MIN_MATCH - 1;
                self.strstart += 1;
            } else if self.match_available {
                let c = self.window[self.strstart - 1];
                self.tally_lit(c);
                self.strstart += 1;
                self.lookahead -= 1;
            } else {
                self.match_available = true;
                self.strstart += 1;
                self.lookahead -= 1;
            }
        }
        if self.match_available {
            let c = self.window[self.strstart - 1];
            self.tally_lit(c);
            self.match_available = false;
        }
        self.flush_block(true);
    }
}

/// zlib.compressobj(9, zlib.DEFLATED, -15): compress(data) + flush(). data must be under 32 KB.
pub fn deflate_raw(data: &[u8]) -> Vec<u8> {
    assert!(data.len() < MAX_DIST, "deflate_raw is for small inputs (one window)");
    let mut window = vec![0u8; 2 * W_SIZE];
    window[..data.len()].copy_from_slice(data);
    let mut d = Deflate {
        window,
        prev: vec![0; W_SIZE],
        head: vec![0; HASH_SIZE],
        ins_h: 0,
        strstart: 0,
        lookahead: 0,
        block_start: 0,
        match_length: MIN_MATCH - 1,
        prev_length: MIN_MATCH - 1,
        prev_match: 0,
        match_start: 0,
        match_available: false,
        dyn_ltree: [Ct::default(); HEAP_SIZE],
        dyn_dtree: [Ct::default(); 2 * D_CODES + 1],
        bl_tree: [Ct::default(); 2 * BL_CODES + 1],
        l_max_code: 0,
        d_max_code: 0,
        bl_max_code: 0,
        bl_count: [0; MAX_BITS + 1],
        heap: [0; 2 * L_CODES + 1],
        heap_len: 0,
        heap_max: 0,
        depth: [0; 2 * L_CODES + 1],
        syms: Vec::new(),
        opt_len: 0,
        static_len: 0,
        bi_buf: 0,
        bi_valid: 0,
        out: Vec::new(),
    };
    d.init_block();
    d.run(data.len());
    d.out
}

// ------------------------------------------------------------------------------------------------ inflate
/// Why inflate stopped (zlib.error, in Python).
#[derive(Debug, PartialEq)]
pub struct InflateError(pub &'static str);

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    bit: u32,
    nbits: u32,
}

impl Bits<'_> {
    fn need(&mut self, n: u32) -> Result<(), InflateError> {
        while self.nbits < n {
            if self.pos >= self.data.len() {
                return Err(InflateError("incomplete or truncated stream"));
            }
            self.bit |= (self.data[self.pos] as u32) << self.nbits;
            self.pos += 1;
            self.nbits += 8;
        }
        Ok(())
    }
    fn bits(&mut self, n: u32) -> Result<u32, InflateError> {
        if n == 0 {
            return Ok(0);
        }
        self.need(n)?;
        let v = self.bit & ((1u32 << n) - 1);
        self.bit >>= n;
        self.nbits -= n;
        Ok(v)
    }
}

/// A canonical Huffman decoding table: counts per length and symbols in order.
struct Huff {
    count: [u16; MAX_BITS + 1],
    symbol: Vec<u16>,
}

/// zlib's inftrees rules. Over-subscribed sets are errors, and so are incomplete ones, except a single code of
/// length 1.
fn huff(lengths: &[u8], err: &'static str, code_lengths: bool) -> Result<Huff, InflateError> {
    let mut count = [0u16; MAX_BITS + 1];
    for &l in lengths {
        count[l as usize] += 1;
    }
    let mut max = MAX_BITS;
    while max >= 1 && count[max] == 0 {
        max -= 1;
    }
    let mut offs = [0u16; MAX_BITS + 2];
    if max == 0 {
        // zlib accepts no codes, for a stream that never uses this table
        return Ok(Huff { count, symbol: Vec::new() });
    }
    let mut left: i32 = 1;
    for len in 1..=MAX_BITS {
        left <<= 1;
        left -= count[len] as i32;
        if left < 0 {
            return Err(InflateError(err));
        }
    }
    if left > 0 && (code_lengths || max != 1) {
        return Err(InflateError(err));
    }
    for len in 1..MAX_BITS {
        offs[len + 1] = offs[len] + count[len];
    }
    let mut symbol = vec![0u16; lengths.len()];
    for (sym, &l) in lengths.iter().enumerate() {
        if l != 0 {
            symbol[offs[l as usize] as usize] = sym as u16;
            offs[l as usize] += 1;
        }
    }
    Ok(Huff { count, symbol })
}

fn decode_sym(b: &mut Bits, h: &Huff, bad: &'static str) -> Result<u16, InflateError> {
    let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
    for len in 1..=MAX_BITS {
        code |= b.bits(1)? as i32;
        let count = h.count[len] as i32;
        if code - count < first {
            return Ok(h.symbol[(index + (code - first)) as usize]);
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    Err(InflateError(bad))
}

const LBASE: [u16; 29] =
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const DBASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];

fn codes(b: &mut Bits, out: &mut Vec<u8>, lit: &Huff, dist: &Huff) -> Result<(), InflateError> {
    loop {
        let sym = decode_sym(b, lit, "invalid literal/length code")? as usize;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            return Ok(());
        } else {
            let sym = sym - 257;
            if sym >= 29 {
                return Err(InflateError("invalid literal/length code"));
            }
            let len = LBASE[sym] as usize + b.bits(EXTRA_LBITS[sym] as u32)? as usize;
            let ds = decode_sym(b, dist, "invalid distance code")? as usize;
            if ds >= 30 {
                return Err(InflateError("invalid distance code"));
            }
            let d = DBASE[ds] as usize + b.bits(EXTRA_DBITS[ds] as u32)? as usize;
            if d > out.len() {
                return Err(InflateError("invalid distance too far back"));
            }
            for _ in 0..len {
                out.push(out[out.len() - d]);
            }
        }
    }
}

/// zlib.decompress(data, -15), raw deflate. Anything after the last block is ignored, as Python does.
pub fn inflate_raw(data: &[u8]) -> Result<Vec<u8>, InflateError> {
    let mut b = Bits { data, pos: 0, bit: 0, nbits: 0 };
    let mut out = Vec::new();
    loop {
        let last = b.bits(1)?;
        match b.bits(2)? {
            0 => {
                b.bit = 0;
                b.nbits = 0;
                if b.pos + 4 > data.len() {
                    return Err(InflateError("incomplete or truncated stream"));
                }
                let len = u16::from_le_bytes([data[b.pos], data[b.pos + 1]]) as usize;
                let nlen = u16::from_le_bytes([data[b.pos + 2], data[b.pos + 3]]) as usize;
                if len != (!nlen & 0xffff) {
                    return Err(InflateError("invalid stored block lengths"));
                }
                b.pos += 4;
                if b.pos + len > data.len() {
                    return Err(InflateError("incomplete or truncated stream"));
                }
                out.extend_from_slice(&data[b.pos..b.pos + len]);
                b.pos += len;
            }
            1 => {
                let mut l = [0u8; 288];
                l[..144].fill(8);
                l[144..256].fill(9);
                l[256..280].fill(7);
                l[280..].fill(8);
                let lit = huff(&l, "invalid literal/lengths set", false)?;
                let dist = huff(&[5u8; 32], "invalid distances set", false)?; // 30 and 31 exist and are invalid
                codes(&mut b, &mut out, &lit, &dist)?;
            }
            2 => {
                let nlen = b.bits(5)? as usize + 257;
                let ndist = b.bits(5)? as usize + 1;
                let ncode = b.bits(4)? as usize + 4;
                if nlen > 286 || ndist > 30 {
                    return Err(InflateError("too many length or distance symbols"));
                }
                let mut lens = [0u8; 19];
                for &o in BL_ORDER.iter().take(ncode) {
                    lens[o] = b.bits(3)? as u8;
                }
                let lencode = huff(&lens, "invalid code lengths set", true)?;
                if lencode.symbol.is_empty() {
                    return Err(InflateError("invalid code lengths set"));
                }
                let mut all = vec![0u8; nlen + ndist];
                let mut i = 0;
                while i < nlen + ndist {
                    let sym = decode_sym(&mut b, &lencode, "invalid code lengths set")?;
                    if sym < 16 {
                        all[i] = sym as u8;
                        i += 1;
                    } else {
                        let (val, rep) = match sym {
                            16 => {
                                if i == 0 {
                                    return Err(InflateError("invalid bit length repeat"));
                                }
                                (all[i - 1], 3 + b.bits(2)? as usize)
                            }
                            17 => (0, 3 + b.bits(3)? as usize),
                            _ => (0, 11 + b.bits(7)? as usize),
                        };
                        if i + rep > nlen + ndist {
                            return Err(InflateError("invalid bit length repeat"));
                        }
                        all[i..i + rep].fill(val);
                        i += rep;
                    }
                }
                if all[256] == 0 {
                    return Err(InflateError("invalid code -- missing end-of-block"));
                }
                let lit = huff(&all[..nlen], "invalid literal/lengths set", false)?;
                let dist = huff(&all[nlen..], "invalid distances set", false)?;
                codes(&mut b, &mut out, &lit, &dist)?;
            }
            _ => return Err(InflateError("invalid block type")),
        }
        if last == 1 {
            return Ok(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn deflate_round_trips() {
        let mut seed = 12345u32;
        for n in [0usize, 1, 2, 3, 10, 100, 1000, 3000] {
            let data: Vec<u8> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    b"abcab\x00\x01"[(seed >> 16) as usize % 7]
                })
                .collect();
            assert_eq!(inflate_raw(&deflate_raw(&data)).unwrap(), data, "{n}");
        }
    }
}
