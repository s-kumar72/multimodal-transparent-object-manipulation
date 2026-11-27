"""
Read-out of the repeated pilot of task T1 (results/hard_scenario/bench_v2/README.md, "The repeated pilot"), from the
episodes.json files of the final evaluations (no new runs):

  python policy/readout_v2_pilot_b.py > results/hard_scenario/bench_v2/readout_pilot_b.txt

The read-out of the pilot (policy/readout_v2_pilot.py), unchanged, on the level with the thermal states of point 3 of
the rule (V2b_place) and, side by side, on the level of the first pilot (V2_place): per level the success per cell of
the four runs, the two stops, `cue` against plain rgbdt and against the one network on the same episodes.
"""
import readout_v2_pilot as R

R.LEVELS = {"V2b_place": "thermal faint 10 to 26 noise units, shows 26 to 103 (point 3 of the rule)",
            "V2_place": "thermal faint 3 to 10, shows 10 to 103 (the first pilot)"}

if __name__ == "__main__":
    R.main()
