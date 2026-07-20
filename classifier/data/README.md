# Local classifier inputs

This directory intentionally excludes raw dialogue text from Git. Before
training, place or generate these files locally:

| File | Rows | SHA-256 / source |
|---|---:|---|
| `swda_train.csv` | 179,766 | `dc28113d1068b0579a7c7eaec85d02a0343bb340a51a420798e7e80913b46c05` |
| `swda_val.csv` | 19,974 | `b2164a396318302bf3868e29b2f918b985e0125a4b36ef236073e00610a4308b` |
| `studychat.csv` | 16,851 | generated from the official gated StudyChat revision documented below |

The fixed SwDA CSVs contain Switchboard transcripts and are not redistributed
by this public repository. Copy your authorized local copies into this
directory. `validate_inputs.py` rejects any file whose checksum differs from
the inputs used in the documented experiment.

Generate `studychat.csv` after accepting the official dataset access terms and
authenticating with Hugging Face:

```bash
../.venv/bin/python ../download_studychat.py
```

The downloader fixes revision
`24d7987d9fbb30d9da12acc53455a10f1cdd2d7f` of
`wmcnicho/StudyChat` and keeps only the columns required by the classifier and
downstream analysis.
