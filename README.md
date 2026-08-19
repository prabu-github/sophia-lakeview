# sophia-lakeview
Code for sophia-lakeview

---

# Data setup: Data download and make compatible

- `cd projworks`
- `python sophia-lakeview/datasetup.py --extension csv`
- `python sophia-lakeview/datasetup.py --extension parquet`
- CSV is for human readability, parquet is for internal use! 


---

# Test `project.py`
- Data setup assumed done.
- `cd projworks`
- `python sophia-lakeview/test_project.py --get_xtransforms`
- `python sophia-lakeview/test_project.py --get_ytransforms`
- `python sophia-lakeview/test_project.py --get_modelnames --pattern '*'`
- `python sophia-lakeview/test_project.py --get_data --modelname dplsr__sophia260424-ammonium__vsbl-uv__id`

---

# CHTC setup: Create submit files

- `cd projworks`
- `python sophia-lakeview/chtcsetup.py --models_per_submit 1`

---

# Note for `run.py`
- `cd projworks`
- Two ways to specify `--train_model`:
   - `python sophia-lakeview/run.py --train_model MODELNAME`
   - `python sophia-lakeview/run.py --train_model MODELNAME1:MODELNAME2:MODELNAME3:...`
    
---

# CHTC

### local
- `cd projworks`
- `tar --exclude hytraits/test -czf  forchtc/hytraits.tar.gz hytraits/`
- `tar --exclude sophia-lakeview/assets --exclude sophia-lakeview/origdata --exclude sophia-lakeview/gendata -czf forchtc/sophia-lakeview.tar.gz sophia-lakeview`

### staging
- `hytraits-cpu.sif` must be in `sophia-lakeview/toremote`
- Clean `sophia-lakeview/fromremote` as needed!

### submit
- Clean `/home/pravindran/sophia-lakeview`

### local to staging
- `cd projworks`
- `scp forchtc/hytraits.tar.gz forchtc/sophia-lakeview.tar.gz pravindran@transfer.chtc.wisc.edu:/staging/p/pravindran/sophia-lakeview/toremote`

### local to submit
- `cd projworks`
- `scp sophia-lakeview/assets/chtc_run.sh sophia-lakeview/gendata/forchtc/* pravindran@townsend-ap.chtc.wisc.edu:/home/pravindran/sophia-lakeview`


### staging to local
- `cd projworks`
- `scp pravindran@transfer.chtc.wisc.edu:/staging/p/pravindran/sophia-lakeview/fromremote/<PATTERN>.tar.gz .`

### Extract tar.gzs
- `for file in *.tar.gz; do tar -xzf "$file"; done`
---






