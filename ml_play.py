"""
The template of the script for the machine learning process in game pingpong
"""

# Import the necessary modules and classes
from mlgame.communication import ml as comm
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def ml_loop(side: str):
    """
    The main loop for the machine learning process

    The `side` parameter can be used for switch the code for either of both sides,
    so you can write the code for both sides in the same script. Such as:
    ```python
    if side == "1P":
        ml_loop_for_1P()
    else:
        ml_loop_for_2P()
    ```

    @param side The side which this script is executed for. Either "1P" or "2P".
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here
    ball_served = False

    # 2. Inform the game process that ml process is ready
    comm.ml_ready()
    c=0
    X=[]
    Y=[]
    
    dis=100
    # 3. Start an endless loop
    while True:
        # 3.1. Receive the scene information sent from the game process
        scene_info = comm.recv_from_game()
        mid=scene_info["platform_1P"][0]+20
        if c==0:
            with open("Predict.pickle","rb") as f:
                knn=pickle.load(f)
            with open("T0.pickle","rb") as f:
                knn0=pickle.load(f)
            with open("T1.pickle","rb") as f:
                knn1=pickle.load(f)
            with open("T2.pickle","rb") as f:
                knn2=pickle.load(f)
            with open("T3.pickle","rb") as f:
                knn3=pickle.load(f)
            with open("T4.pickle","rb") as f:
                knn4=pickle.load(f)
            with open("T5.pickle","rb") as f:
                knn5=pickle.load(f)
            box_px=-1
            c=1 
        box_x=scene_info["blocker"][0]
        if scene_info["frame"]>2:
            if box_x>box_px:
                s=1
            if box_x<box_px:
                s=-1

            feature=[]
            feature.append(scene_info["ball"][0])
            feature.append(scene_info["ball"][1])
            feature.append(scene_info["ball_speed"][0])
            feature.append(scene_info["ball_speed"][1])
            feature.append(scene_info["blocker"][0])
            feature.append(s)
            feature = np.array(feature)
            feature = feature.reshape((-1,6))
            t=knn.predict(feature)

            feature1=[]
            feature1.append(scene_info["ball"][0])
            feature1.append(scene_info["ball"][1])
            feature1.append(scene_info["ball_speed"][0])
            feature1.append(scene_info["ball_speed"][1])
            feature1 = np.array(feature1)
            feature1 = feature1.reshape((-1,4))
            if t==0:
               dis=knn0.predict(feature1)
            if t==1:
                dis=knn1.predict(feature)
            if t==2:
                dis=knn2.predict(feature)
            if t==3:
                dis=knn3.predict(feature)
            if t==4:
                dis=knn4.predict(feature)
            if t==5:
                dis=knn5.predict(feature1) 
            if dis<20:
                dis=20
            if dis>180:
                dis=180
        box_px=box_x
        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.
        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        # 3.3 Put the code here to handle the scene information

        # 3.4 Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
            ball_served = True
        else:
            if dis<mid-7:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
            elif dis>mid+7:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
            else :
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
