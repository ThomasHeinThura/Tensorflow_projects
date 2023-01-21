# Introduction before you test out. 
* These are Tensorflow small projects for practice. These projects-codes are for those who want to test some projects without wanting to tinker and for those who want to compare the results. 
* These projects are tested on an AMD Ryzen3 5425U laptop, 16gb of ram and Desktop Nvidia 3060 12gb GPU running in Arcolinux (linux-kernal 6.1) OS and CPU governor `performance`. So, time results may vary from tested resources to resources (CPU, GPU, OS and power state.).
* If laptops are in power save mode, this takes 2x slower than performance mode.
*  There are some projects which need to download datasets from Kaggle or Tensorflow Dataset and import them on your own. Lists of datasets that need to download are :
    > 1. flower dataset
    > 2. sign minst
    > 3. BBC dataset
    > 4. disater dataset 
    > 5. sarasm dataset  
* For those who want to make models. This is best practice to use `Jupyter notebook` at first. Because you can adjust and check for preparing data, input-output shape, and models. Stand-alone python files are for a direct run. 
*  In some folders, there are readme files for showing results, difficulties, opinions, further study and future improvement. 
* In some models, I train with functional API. 
*  There is a small regression dataset. This can be compared with the Time series dataset. The difference between normal regression is making windows and horizons (past and future). 