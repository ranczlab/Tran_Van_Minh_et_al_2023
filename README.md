### Data and code for Tran Van Minh et al PLOS One paper. 
- for your own data, follow input dataformat in data/raw/
- jupyter notebooks step1_ to step8_ reproduce the analysis (already run in the repo, if you only want to reproduce figures, you can skip to that below)
- jupyter notebook step9 recreates stimulation data for figS20
- jupyter notebooks in figures/plot_figXX reproduce figures
- tested on macOS with Python 3.9 and respective packages (except seaborn==0.11.1)

'code'
conda create --name rabies2023 python=3.9
conda activate rabies2023
conda install -c anaconda ipykernel --y
conda install -c conda-forge lmfit --y 
conda install matplotlib pandas pytables scikit-learn seaborn==0.11.1 --y