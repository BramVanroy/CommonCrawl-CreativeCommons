# Creative Commons License annotation of CommonCrawl

## Running

- CC-MAIN-2024-51
- CC-MAIN-2023-06

## To do


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
sbatch upload_all.slurm -d /dodrio/scratch/projects/2024_107/gpt-nl-copyright/output/CC-MAIN-2024-51 -w 15 -e 30
```