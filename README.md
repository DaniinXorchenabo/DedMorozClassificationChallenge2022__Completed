

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
   
   [https://drive.google.com/file/d/1KsotujU1g2Yi2ivbUJuwo5fPWq_pBwzD/view?usp=sharing](https://drive.google.com/file/d/1KsotujU1g2Yi2ivbUJuwo5fPWq_pBwzD/view?usp=sharing)
   
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
   
      [https://drive.google.com/file/d/1KsotujU1g2Yi2ivbUJuwo5fPWq_pBwzD/view?usp=sharing](https://drive.google.com/file/d/1KsotujU1g2Yi2ivbUJuwo5fPWq_pBwzD/view?usp=sharing)
   
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

# Results

## testing data
```
             | w                          h
             | h                          i
             | i                          p              r
             | t                          p              h
             | e                 a     e  o  k           i
             | _                 n     l  p  a           n
             | h        t        t  w  e  o  n     d     o
             | o     z  u  h  s  e  i  p  t  g  b  o     c
             | r  d  e  r  o  h  l  n  h  a  a  i  n     e
             | s  e  b  t  r  e  o  t  a  m  r  s  k  c  r
             | e  e  r  l  s  e  p  e  n  u  o  o  e  o  o
             | s  r  a  e  e  p  e  r  t  s  o  n  y  w  s
___________________________________________________________
white_horses | 8  0  0  0  0  0  0  0  0  0  0  0  0  0  0
-----------------------------------------------------------
deer         | 0  7  0  0  0  0  1  0  0  0  0  0  0  0  0
-----------------------------------------------------------
zebra        | 0  0  7  0  0  0  0  0  0  0  0  0  0  0  0
-----------------------------------------------------------
turtle       | 0  0  0  7  0  0  0  0  0  0  0  0  0  0  0
-----------------------------------------------------------
horse        | 0  0  0  0  8  0  0  0  0  0  0  0  0  0  0
-----------------------------------------------------------
sheep        | 0  0  0  0  0  6  0  0  0  0  0  1  0  0  0
-----------------------------------------------------------
antelope     | 0  0  0  0  0  0  7  0  0  0  0  0  0  0  0
-----------------------------------------------------------
winter       | 0  0  0  0  0  0  0  5  0  0  0  0  0  0  0
-----------------------------------------------------------
elephant     | 0  0  0  0  0  0  0  0  7  0  0  0  0  0  0
-----------------------------------------------------------
hippopotamus | 0  0  0  0  0  0  0  0  0  8  0  0  0  0  0
-----------------------------------------------------------
kangaroo     | 0  0  0  0  0  0  0  0  0  0  7  0  0  0  0
-----------------------------------------------------------
bison        | 0  0  0  0  0  0  0  0  0  0  0  7  0  0  0
-----------------------------------------------------------
donkey       | 0  0  0  0  0  0  0  0  0  0  0  0  6  0  0
-----------------------------------------------------------
cow          | 0  0  0  0  0  0  0  0  0  0  0  0  0  7  0
-----------------------------------------------------------
rhinoceros   | 0  0  0  0  0  0  0  0  0  0  0  0  0  0  7
-----------------------------------------------------------
Validation loss: 0.101
Validation acc:  96.218
White horse precision metric: 1.000
```

## training data

```
             |                   w     h
             |                   h     i
             |          r        i     p
             |          h        t     p
             |       e  i     k  e     o  a
             |       l  n     a  _     p  n
             |       e  o     n  h  t  o  t        w  d
             | h  z  p  c  s  g  o  u  t  e  b     i  o
             | o  e  h  e  h  a  r  r  a  l  i     n  n  d
             | r  b  a  r  e  r  s  t  m  o  s  c  t  k  e
             | s  r  n  o  e  o  e  l  u  p  o  o  e  e  e
             | e  a  t  s  p  o  s  e  s  e  n  w  r  y  r
___________________________________________________________
horse        | 67 0  0  0  0  0  0  0  0  0  0  0  0  0  0
-----------------------------------------------------------
zebra        | 0  64 0  0  0  0  0  0  0  0  0  0  0  0  0
-----------------------------------------------------------
elephant     | 0  0  63 0  0  1  0  0  0  0  0  0  0  1  0
-----------------------------------------------------------
rhinoceros   | 0  0  0  63 0  0  0  0  0  0  0  0  0  0  0
-----------------------------------------------------------
sheep        | 0  0  0  0  59 0  0  0  0  0  0  0  0  1  0
-----------------------------------------------------------
kangaroo     | 0  0  0  0  0  61 0  0  0  0  0  0  0  0  0
-----------------------------------------------------------
white_horses | 0  0  0  0  0  0  63 0  0  0  0  0  0  0  0
-----------------------------------------------------------
turtle       | 0  0  0  0  0  0  0  63 0  0  0  0  0  0  0
-----------------------------------------------------------
hippopotamus | 0  0  0  0  0  0  0  0  62 0  0  0  0  0  0
-----------------------------------------------------------
antelope     | 0  0  0  0  0  0  0  0  0  62 0  0  0  0  0
-----------------------------------------------------------
bison        | 0  0  0  0  0  0  0  0  0  0  65 0  0  0  0
-----------------------------------------------------------
cow          | 0  0  0  0  0  0  0  0  0  0  0  62 0  0  0
-----------------------------------------------------------
winter       | 0  0  0  0  0  0  0  0  0  0  0  0  65 0  0
-----------------------------------------------------------
donkey       | 0  0  0  0  0  0  0  0  0  0  0  0  0  63 0
-----------------------------------------------------------
deer         | 0  0  0  0  0  0  0  0  0  0  0  0  0  0  58
-----------------------------------------------------------
Training loss: 0.011,
Training acc:  99.813
White horse precision metric: 1.000
```

## All data

```
|              |                                     w  h
|              |                                     h  i
|              |          r                          i  p 
|              |          h                          t  p
|              |       k  i           a        e     e  o
|              |       a  n           n        l     _  p
|              |       n  o     t     t     d  e     h  o
|              | z  b  g  c     u     e  h  o  p  s  o  t
|              | e  i  a  e  d  r     l  o  n  h  h  r  a
|              | b  s  r  r  e  t  c  o  r  k  a  e  s  m
|              | r  o  o  o  e  l  o  p  s  e  n  e  e  u
|              | a  n  o  s  r  e  w  e  e  y  t  p  s  s
| ________________________________________________________
| zebra        |105 0  0  0  0  0  0  0  0  0  0  0  0  0
| --------------------------------------------------------
| bison        | 0 105 0  0  0  0  0  0  0  0  0  0  0  0
| --------------------------------------------------------
| kangaroo     | 0  0 102 0  0  0  0  0  0  0  0  0  0  0 
| --------------------------------------------------------
| rhinoceros   | 0  0  0 103 0  0  0  0  0  0  0  0  0  0
| --------------------------------------------------------
| deer         | 0  0  1  0  94 0  0  0  0  0  0  0  0  0
| --------------------------------------------------------
| turtle       | 0  0  0  0  0  93 0  0  0  0  0  0  0  0
| --------------------------------------------------------
| cow          | 0  0  0  0  0  0 102 0  0  0  0  0  0  0
| --------------------------------------------------------
| antelope     | 0  0  0  0  0  0  0  98 0  0  0  0  0  0
| --------------------------------------------------------
| horse        | 0  0  0  0  1  0  0  0 101 0  0  0  0  0
| --------------------------------------------------------
| donkey       | 0  0  0  0  0  0  0  0  0 104 0  0  0  0 
| --------------------------------------------------------
| elephant     | 0  0  0  0  0  0  0  0  0  0  99 0  0  0
| --------------------------------------------------------
| sheep        | 0  0  0  0  0  0  0  0  0  0  0 105 0  0
| --------------------------------------------------------
| white_horses | 0  0  0  0  0  0  0  0  1  0  0  0  99 0
| --------------------------------------------------------
| hippopotamus | 0  0  0  0  0  0  0  0  0  0  0  0  0 101
| --------------------------------------------------------
| Validation loss: 0.010, validation acc: 99.730
| White horse precision metric: 1.000000
| Validation loss: 0.010
| Validation acc: 99.730
| White horse precision metric: 1.000000
```

## training graph

### Accuracy

![accuracy_graph](https://user-images.githubusercontent.com/45897837/168446896-114bd03a-e4b7-4c13-9b2f-d2e651d5da39.png)


### Loss
![loss_graph](https://user-images.githubusercontent.com/45897837/168446899-df7f85a7-2ad7-4a74-9b35-248a813ca389.png)
