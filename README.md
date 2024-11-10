# wuRenji

Configs fo training are in `configs`.

How to use:

`sh scripts/[_model you choose_].sh`

To test, add weight path to shellscript.

Github repo: https://github.com/VielyNi/wuRenji

Logs and scores are in `logs`, and ckpts are in `ckpts`.(CTR endwith gcl)

we have 5 models(ctrgcn, tdgcn, mixformer, stt, msst) in joint and bone, so you will get 10 scores.
Use `vm.py` to get the final score.

Remember to name the dirctorys as :
- ctrj, ctrb, tdj, tdb, mixj, mixb, sttj, sttb, msstj, msstb

we optain the alpha weight using search.


