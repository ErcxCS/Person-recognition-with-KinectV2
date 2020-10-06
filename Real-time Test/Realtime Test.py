# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:58:23 2020

@author: berke
"""

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

import pygame
import ctypes
import sys

import numpy as np

import pandas as pd
import math

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread
    
from sklearn.externals import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import  accuracy_score


excel_dataset = "Dataset without IMG info.xlsx"
exclude_header_indexes = [3, 15, 19, 21, 22, 23, 24]
exclude_attribute_indexes = [3, 4, 5, 6]
 
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                  pygame.color.THECOLORS["blue"], 
                  pygame.color.THECOLORS["green"], 
                  pygame.color.THECOLORS["orange"], 
                  pygame.color.THECOLORS["purple"], 
                  pygame.color.THECOLORS["yellow"], 
                  pygame.color.THECOLORS["violet"]]

class KinectRuntime(object):
    def __init__(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color
                                                       | PyKinectV2.FrameSourceTypes_Body 
                                                       | PyKinectV2.FrameSourceTypes_Depth)
        
        self._bodies = None
        self._done = False
        self._dframes = []
        self._counter = 0

        self._width_color_frame = self._kinect.color_frame_desc.Width
        self._height_color_frame = self._kinect.color_frame_desc.Height
        self._width_depth_frame = self._kinect.depth_frame_desc.Width 
        self._height_depth_frame = self._kinect.depth_frame_desc.Height
        
        pygame.init()
        pygame.display.set_caption("Real-time Test")
        self._clock = pygame.time.Clock()
        self._infoObject = pygame.display.Info()
        self._frame_surface = pygame.Surface((self._width_color_frame, self._height_color_frame), 0, 32)
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1),
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        
        CSP_array = PyKinectV2._CameraSpacePoint * (self._width_depth_frame * self._height_depth_frame)
        self._depth_to_CSP_buffer = CSP_array()
        
        self._kisiler = []
        self._encoder = LabelEncoder()
        
        self._kisi = 23
        self._yTrue = []
        self._voting_pred = []
        self._votting_model = []
        
        model_type = "FOLD_"
        self._randomforest_model, \
            self._gradientboosting_model, \
                self._xgboost_model, \
                    self._knearestneighbors_model, \
                        self._artificialneuralnetwork_model, \
                            self._deepneuralnetwork_model = self.load_model(model_type)
        
        self._classifier_names = ["Random Forest", "Gradient Boosting", "Extreme Gradient Boosting", "K-Nearest Neighbors", "Artificial Neural Network", "Deep Neural Network"]
        self._model_preds = [0] * len(self._classifier_names)
        self._temp_preds = [0] * len(self._classifier_names)
        self._voting_model_preds = [0] * len(self._classifier_names)
        for z in range(len(self._classifier_names)):
            self._model_preds[z] = []
            self._temp_preds[z] = []
            self._voting_model_preds[z] = []
        

    def load_model(self, typ=""):
        model_path = "models/"
        typ = model_path + typ
        rf_model = joblib.load(typ+"rf_clf.pkl")
        gb_model = joblib.load(typ+"gb_clf.pkl")
        xgb_model = joblib.load(typ+"xgb_clf.pkl")
        knn_model = joblib.load(typ+"knn_clf.pkl")
        ann_model = joblib.load(typ+"mlp_clf.pkl")
        dnn_model = tf.keras.models.load_model(typ+"dnn_clf.model")
        
        dataset = pd.read_excel(excel_dataset)
        y = dataset.iloc[:, -1].values
        self._encoder.fit(y)
        
        for i in range(len(y)):
            if y[i] not in self._kisiler:
                self._kisiler.append(y[i])
            
        
        return rf_model, gb_model, xgb_model, knn_model, ann_model, dnn_model
    
    def make_predictions(self, data):
        
        X = data.iloc[:,:].values
        
        rf_pred = self._randomforest_model.predict(X)
        gb_pred = self._gradientboosting_model.predict(X)
        xgb_pred = self._xgboost_model.predict(X)
        knn_pred = self._knearestneighbors_model.predict(X)
        ann_pred = self._artificialneuralnetwork_model.predict(X)
        dnn_preds = self._deepneuralnetwork_model.predict(X)
        
        dnn_pred = []
        for i in range(len(dnn_preds)):
            dnn_pred.append(np.argmax(dnn_preds[i]))
            
        dnn_pred = self._encoder.inverse_transform(dnn_pred)
        
        preds = [rf_pred, gb_pred, xgb_pred, knn_pred, ann_pred, dnn_pred]
        self._yTrue.append(self._kisi)
        
        return preds
    
    def mapDepthFrame2CSP(self, depth_frame):
        length = self._width_depth_frame * self._height_depth_frame
        assert length == depth_frame.size
        ptr_depth = np.ctypeslib.as_ctypes(depth_frame.flatten())
        error_state = self._kinect._mapper.MapDepthFrameToCameraSpace(
                                                                length, ptr_depth,
                                                                length, self._depth_to_CSP_buffer)

        if error_state:
            raise "Could not map depth frame to camera space! " + str(error_state)

        pf_CSPs = ctypes.cast(self._depth_to_CSP_buffer, ctypes.POINTER(ctypes.c_float))
        data = np.ctypeslib.as_array(pf_CSPs, shape=(self._height_depth_frame, self._width_depth_frame, 3))
        
        del pf_CSPs, ptr_depth, depth_frame
        return np.copy(data)
    
    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return
        
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)
        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)  
        except:
            pass
        
    def drawBody(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);
    
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # Left Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);
        
    def drawColorFrame(self, RGB_frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, RGB_frame.ctypes.data, RGB_frame.size)
        del address
        target_surface.unlock()
    
    def getRGBframeInfo(self, RGB_joint_points):
        RGB_joints_data = []
        for joint in list(RGB_joint_points):
            X = round(joint.x)
            Y = round(joint.y)
                
            RGB_joints_data.append((X, Y))
        return RGB_joints_data
    
    def getRealJointCoordinates(self, depth_frame, depth_joint_points):
        CSP = self.mapDepthFrame2CSP(depth_frame) 
        real_joints_data = []
        for joint in list(depth_joint_points):
            X = CSP[round(joint.y)][round(joint.x)][0]
            Y = CSP[round(joint.y)][round(joint.x)][1]
            Z = CSP[round(joint.y)][round(joint.x)][2]
            real_joints_data.append((X, Y, Z))
        return real_joints_data
    
    def getJointQuaternions(self, orientations):        
        quats = []        
        for joint in range(PyKinectV2.JointType_Count):            
            Qw = orientations[joint].Orientation.w
            Qx = orientations[joint].Orientation.x
            Qy = orientations[joint].Orientation.y
            Qz = orientations[joint].Orientation.z
            quats.append((Qw, Qx, Qy, Qz))
            
        return quats
    
    def getQuatsVectors(self, quaternions):
        quatsVectors = []
        for joint in range(PyKinectV2.JointType_Count):
            Quat = quaternions[joint]
            
            Qw = Quat[0]
            Qx = Quat[1]
            Qy = Quat[2]
            Qz = Quat[3]
            
            W = math.degrees(math.acos(Qw)) * 2
            X = Qx / math.sin(math.radians(W/2))
            Y = Qy / math.sin(math.radians(W/2))
            Z = Qz / math.sin(math.radians(W/2))
            
            quatsVectors.append((W, X, Y, Z))
        return quatsVectors
    
    def getDFinfo(self, real_joints_data, orientation_vectors):
        rowInfo = []
        
        for joint in range(PyKinectV2.JointType_Count):

            rowInfo.append(real_joints_data[joint][0])
            rowInfo.append(real_joints_data[joint][1])
            rowInfo.append(real_joints_data[joint][2])
            
            if joint not in exclude_header_indexes:
                rowInfo.append(orientation_vectors[joint][0])
                rowInfo.append(orientation_vectors[joint][1])
                rowInfo.append(orientation_vectors[joint][2])
                rowInfo.append(orientation_vectors[joint][3])
        
        excel_row = []
        excel_row.append(rowInfo)
        DF = pd.DataFrame(excel_row)
        
        return DF
        
    
    def PredictionVoting(self, predictions):
        kisi_secim = [0] * len(self._kisiler)
        
        for k in range(len(self._classifier_names)):
            for j in range(len(self._kisiler)):
                if predictions[k][0] == self._kisiler[j]:
                    kisi_secim[j] += 1
        
        kisi_idx = kisi_secim.index(max(kisi_secim))
        combined_prediction = self._kisiler[kisi_idx]
        
        """print("############################################")
        for j in range(len(self._classifier_names)):
            print(predictions[j][0], self._classifier_names[j])"""
        
        return combined_prediction
    
    def ModelVoting(self, predictions):

        model_predictions = []
        for k in range(len(self._classifier_names)):
            kisi_secim = [0] * len(self._kisiler)
            
            for i in range(len(predictions[k])):
                for j in range(len(self._kisiler)):
                    if predictions[k][i] == self._kisiler[j]:
                        kisi_secim[j] += 1
            
            kisi_idx = kisi_secim.index(max(kisi_secim))
            model_predictions.append(self._kisiler[kisi_idx])
        
        model_voting = [0] * len(self._kisiler)
        for k in range(len(self._classifier_names)):
            for j in range(len(self._kisiler)):
                if model_predictions[k] == self._kisiler[j]:
                    model_voting[j] += 1
            
            self._voting_model_preds[k].append(model_predictions[k])
        
        kisi_idx = model_voting.index(max(model_voting))
        voted_prediction = self._kisiler[kisi_idx]
        
        return voted_prediction
        
    
    def Runtime(self):
        while not self._done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._done = True

                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

            if self._kinect.has_new_color_frame():
                RGB_frame = self._kinect.get_last_color_frame() # kinectden renkli framei alma
                depth_frame = self._kinect.get_last_depth_frame() # kinectden derinlik frameini alma
                self._bodies = self._kinect.get_last_body_frame() # kinectden vücut bilgilerini alma
                
                self.drawColorFrame(RGB_frame, self._frame_surface)
                if self._bodies is not None: 
                    for i in range(0, self._kinect.max_body_count):
                        body = self._bodies.bodies[i]
                        if body.is_tracked:
                            joints = body.joints # düğümler alındı
                            RGB_joint_points = self._kinect.body_joints_to_color_space(joints) # RGB framei için iskelet düğümleri

                            orientations = body.joint_orientations
                            
                            depth_joint_points = self._kinect.body_joints_to_depth_space(joints) # depth framei için iskelet düğümleri
                            quaternions = self.getJointQuaternions(orientations) # orientationlardan quaternionlar elde edildi

                            real_joints_data = self.getRealJointCoordinates(depth_frame, depth_joint_points) # Depth framei ile düğümler için gerçek X, Y ve Z uzaklık bilgileri
                            orientation_vectors = self.getQuatsVectors(quaternions) # düğümlerin quaternionları ile W açısı ve X,Y,Z vektörleri hesaplandı
                            
                            d_frame = self.getDFinfo(real_joints_data, orientation_vectors)
                            if float('-inf') not in d_frame.values:
                                self._dframes.append(d_frame)
                                predictions = self.make_predictions(d_frame)
                                
                                for j in range(len(self._classifier_names)):
                                    self._model_preds[j].append(predictions[j][0])
                                    self._temp_preds[j].append(predictions[j][0])
                                    
                                    """print(self._classifier_names[j], self._model_preds[j][len(self._dframes) - 1])
                                    print(self._classifier_names[j], " Accuracy = ", accuracy_score(self._yTrue, self._model_preds[j]))
                                print("################################")"""
                                    
                                prediction_voting = self.PredictionVoting(predictions)
                                self._voting_pred.append(prediction_voting)
                                
                                """print("Prediction Voted Prediction = ", prediction_voting)
                                print("Prediction Voted Accuracy = ", accuracy_score(self._yTrue, self._voting_pred))"""
                                
                                #print("Combined = ", prediction_voting)
                                
                                if len(self._dframes) % 15 == 0: # 15 30 45
                                    
                                    model_voting = self.ModelVoting(self._temp_preds)
                                    self._votting_model.append(model_voting)
                                    
                                    y_true = [self._kisi] * len(self._votting_model)
                                    print("Model Voted Prediction = ", model_voting)
                                    print("Model Voting Accuracy = ", accuracy_score(y_true, self._votting_model))
                                    
                                    self._temp_preds = [0] * len(self._classifier_names)
                                    for h in range(len(self._classifier_names)):
                                        self._temp_preds[h] = []
                                        
                                        print(self._classifier_names[j], self._voting_model_preds[j][self._counter])
                                        print("Model Prediction Accuracy = ", accuracy_score(y_true, self._voting_model_preds[j]))
                                    
                                    self._counter += 1

                            self.drawBody(joints, RGB_joint_points, SKELETON_COLORS[1])
                                       

            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0,0))
            surface_to_draw = None
            
            pygame.display.update()
            pygame.display.flip()
            self._clock.tick(15) #FPS

        self._kinect.close()
        pygame.quit()
    
if __name__ == "__main__":

    test = KinectRuntime();
    test.Runtime();

