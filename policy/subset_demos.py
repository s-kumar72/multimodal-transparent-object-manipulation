"""Copies the episodes of chosen visibility cells from a demo set into a new one (same frames, same order).
python policy/subset_demos.py IN.zarr OUT.zarr --cells 0 2 3 6"""
import click
import numpy as np
import zarr


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--cells", required=True, multiple=True, type=int, help="visibility cells to keep (meta/ep_cell)")
def main(src, dst, cells):
    a, b = zarr.open(src, "r"), zarr.open(dst, "w")
    ends = a["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    keep = np.flatnonzero(np.isin(a["meta/ep_cell"][:], cells))
    frames = np.concatenate([np.arange(starts[e], ends[e]) for e in keep])
    data, meta = b.create_group("data"), b.create_group("meta")
    for k, x in a["data"].arrays():
        data.array(k, x[:][frames], chunks=x.chunks, dtype=x.dtype, compressor=x.compressor)
    for k, x in a["meta"].arrays():
        v = np.cumsum((ends - starts)[keep]) if k == "episode_ends" else x[:][keep]
        meta.array(k, v, chunks=x.chunks, dtype=x.dtype, compressor=x.compressor)
    meta.attrs.update(dict(a["meta"].attrs), subset_of=str(src).rsplit("/", 1)[-1], subset_cells=list(cells))
    print(f"{len(keep)} of {len(ends)} episodes, {len(frames)} frames -> {dst}")
    print("cells:", dict(zip(*map(lambda v: v.tolist(), np.unique(a["meta/ep_cell"][:][keep], return_counts=True)))))


if __name__ == "__main__":
    main()
