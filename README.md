

# Run

## Run with `docker`

1. Install a `docker` and `docker-compose`
2. Clone this repo:

```bash
git clone https://github.com/DaniinXorchenabo/DedMorozClassificationChallenge2022__Completed.git
```

3. Go over to a root folder of project:

```bach
cd DedMorozClassificationChallenge2022__Completed
```

4. Download weights
   
   Url for downloading will add in the future.
   
   Move the weights file to `weights` directory and rename to `final_network.pth` 

5. Create a environ variables file:
   1. Copied a `example.env` file and paste as a `.env` file.
   2. Open `.env`
   3. Set `DATA_PATH` as path to validate data
         
       As example, for this file structure, `DATA_PATH=data`
       
       Or this file structure, `DATA_PATH=data/testing`
       
6. Pulling image
   
   ```docker pull daniinxorchenabo/ded_moroz_classification_challenge2022:latest ```

7. Run demonstration script
   ```docker-compose up ```

## Run with python interpreter

1. Clone this repo:

```bash
git clone https://github.com/DaniinXorchenabo/DedMorozClassificationChallenge2022__Completed.git
```

2. Go over to a root folder of project:

```bach
cd DedMorozClassificationChallenge2022__Completed
```

3. Download weights
   
   Url for downloading will add in the future.
   
   Move the weights file to `weights` directory and rename to `final_network.pth` 

4. Create and activate python virtual environment

```bash
python -m venv venv
cd venv/bin/  
source activate  
cd ../..
```

4. Install requirements with `pip`

```bash
pip install --upgrade pip
pip install -r requirements/prod.txt
```

5. Run demonstration script
```bash
python code/demonstration.py
```