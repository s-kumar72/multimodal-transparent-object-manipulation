"""Copies the first 1 / DEN of the episodes of every visibility cell from a demo set into a new one (same frames, same
order), so the smaller set keeps the cell weights of the level (results/hard_scenario/cue_gate/README.md, round 6b).
python policy/subset_demos_frac.py IN.zarr OUT.zarr --den 2"""
import click
import numpy as np
import zarr


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--den", required=True, type=int, help="keep the first 1 / den of every cell's episodes")
def main(src, dst, den):
    a, b = zarr.open(src, "r"), zarr.open(dst, "w")
    ends, cell = a["meta/episode_ends"][:], a["meta/ep_cell"][:]
    starts = np.r_[0, ends[:-1]]
    keep = []
    for c in np.unique(cell):
        idx = np.flatnonzero(cell == c)
        assert len(idx) % den == 0, f"cell {c} has {len(idx)} episodes, not a multiple of {den}"
        keep.append(idx[:len(idx) // den])
    keep = np.sort(np.concatenate(keep))
    frames = np.concatenate([np.arange(starts[e], ends[e]) for e in keep])
    data, meta = b.create_group("data"), b.create_group("meta")
    for k, x in a["data"].arrays():
        data.array(k, x[:][frames], chunks=x.chunks, dtype=x.dtype, compressor=x.compressor)
    for k, x in a["meta"].arrays():
        v = np.cumsum((ends - starts)[keep]) if k == "episode_ends" else x[:][keep]
        meta.array(k, v, chunks=x.chunks, dtype=x.dtype, compressor=x.compressor)
    meta.attrs.update(dict(a["meta"].attrs), subset_of=str(src).rsplit("/", 1)[-1], subset_first_fraction=f"1/{den}")
    print(f"{len(keep)} of {len(ends)} episodes, {len(frames)} frames -> {dst}")
    print("cells:", dict(zip(*map(lambda v: v.tolist(), np.unique(cell[keep], return_counts=True)))))
    print("objects:", dict(zip(*map(lambda v: v.tolist(), np.unique(a["meta/ep_object"][:][keep], return_counts=True)))))


if __name__ == "__main__":
    main()
