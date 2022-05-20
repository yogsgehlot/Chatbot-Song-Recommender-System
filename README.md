This is Chatbot-Song-Recommender-System developed by Yogesh Gehlot (https://github.com/yogsgehlot), Ayush Suryawanshi (https://github.com/AyushSuryawanshi004), Amarpreet Singh (https://github.com/Amarpreet-singh07) and Snehdeep Mehta (https://github.com/Snehdeep31).



1. Download PyCharm from- https://www.jetbrains.com/pycharm/download/#section=windows
2. Download this project and open it in PyCharm.
3. Open terminal of PyCharm and install below libraries.
    i. pip install flask
    ii. pip install chatterbot
    iii. pip install textblob
    iv. pip install ntlk
4.  Run this project.
5. You may face these type problems during run project:-

    problem 1) -  AttributeError : module 'time' has no attribute 'clock'
       
![1 1](https://user-images.githubusercontent.com/75558691/169472505-408019ae-2b3f-493e-b62f-bab1fa0b164a.png)

    solution : 
     Click on last link in terminal. You will get a new open file with name compat.py.
     New open file -
![2 1](https://user-images.githubusercontent.com/75558691/169469986-2be5d523-9b51-4d3d-94e5-fa5181b5ffc0.png)

 Change the statement - "time_func = time.clock" to "time_func = time.process_time() "
   according to below image-

![3 1](https://user-images.githubusercontent.com/75558691/169470691-fb60bb1a-6af7-43b0-bdc0-30d30e2e8eab.png)
 

Run it again .You may get next problem -

   problem 2) -  AttributeError : module 'collections' has no attribute 'Hashtable'
 

![4 1](https://user-images.githubusercontent.com/75558691/169471431-387e97a3-2813-4879-a6a7-08740c3c8d5c.png)

 solution : 
     Click on last link in terminal. You will get a new open file with name constructor.py .
     New open file -

![5 1](https://user-images.githubusercontent.com/75558691/169471593-5acf4052-3554-4f38-ba6b-7576a5a3f1af.png)

Change the statement -  "if not isinstance(key, collections.Hashable):" to "if not isinstance(key, collections.abc.Hashable):"

![6 1](https://user-images.githubusercontent.com/75558691/169472237-00e49eef-7ad9-4678-ae55-f8edefe83982.png)


Run it again.
Now your project is ready.
