# Creative Commons License annotation of CommonCrawl

## Running

- CC-MAIN-2024-51 (started 19/1 @ 13:30)

## To do

- CC-MAIN-2023-06
- CC-MAIN-2020-05
- CC-MAIN-2021-04
- CC-MAIN-2022-05
- CC-MAIN-2019-30

## Commands

Main:

```bash
sbatch launch.slurm CC-MAIN-2024-51
```

Upload:

```bash
# Wait 15 minutes before starting and then upload folder every 30 seconds
sbatch upload_all.slurm -d /dodrio/scratch/projects/2024_107/gpt-nl-copyright/output/ -w 15 -e 30
```
