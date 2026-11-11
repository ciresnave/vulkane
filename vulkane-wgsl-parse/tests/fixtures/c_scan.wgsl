// SPDX-License-Identifier: MIT OR Apache-2.0
enable f16;
requires readonly_and_readwrite_storage_textures;
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
var<workgroup> temp: array<u32, 512>;

fn combine(x: u32, y: u32) -> u32 { return x + y; }

@compute @workgroup_size(256)
fn scan_u32(@builtin(local_invocation_index) li: u32) {
    var offset: u32 = 1u;
    temp[2u * li] = data[2u * li];
    temp[2u * li + 1u] = data[2u * li + 1u];
    for (var d: u32 = 256u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (li < d) {
            let ai = offset * (2u * li + 1u) - 1u;
            let bi = offset * (2u * li + 2u) - 1u;
            temp[bi] = combine(temp[bi], temp[ai]);
        }
        offset *= 2u;
    }
    workgroupBarrier();
    data[2u * li] = temp[2u * li];
    data[2u * li + 1u] = temp[2u * li + 1u];
}
