

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
   
   ![image](https://user-images.githubusercontent.com/45897837/168446365-bfb10d23-8985-4668-bf0c-aee0cb4612b3.png)


5. Create a environ variables file:
   1. Copied a `example.env` file and paste as a `.env` file.
   2. Open `.env`
   3. Set `DATA_PATH` as path to validate data
         
       As example, for this file structure, `DATA_PATH=data`
       
       ![image](https://user-images.githubusercontent.com/45897837/168446539-b757b417-d2ec-4e79-b5c4-ee10a573bdbc.png)

       
       Or this file structure, `DATA_PATH=data/testing`
       
       ![image](https://user-images.githubusercontent.com/45897837/168446467-6af932b0-93fc-4b29-81c5-374e4f25a846.png)

       
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
   
   ![image](https://user-images.githubusercontent.com/45897837/168446365-bfb10d23-8985-4668-bf0c-aee0cb4612b3.png)


4. Create a environ variables file:
   1. Copied a `example.env` file and paste as a `.env` file.
   2. Open `.env`
   3. Set `DATA_PATH` as path to validate data
         
       As example, for this file structure, `DATA_PATH=data`
       
       ![image](https://user-images.githubusercontent.com/45897837/168446539-b757b417-d2ec-4e79-b5c4-ee10a573bdbc.png)

       
       Or this file structure, `DATA_PATH=data/testing`
       
       ![image](https://user-images.githubusercontent.com/45897837/168446467-6af932b0-93fc-4b29-81c5-374e4f25a846.png)

       

5. Create and activate python virtual environment

```bash
python -m venv venv
cd venv/bin/  
source activate  
cd ../..
```

![image](https://user-images.githubusercontent.com/45897837/168446615-9baeb04e-b87b-495d-a019-d1daf24543b2.png)


6. Install requirements with `pip`

```bash
pip install --upgrade pip
pip install -r requirements/prod.txt
```

7. Run demonstration script
```bash
python code/demonstration.py
```
