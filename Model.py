"""
Model definition code. It also contains the list of features used for the task of respiratory deterioration prediction.
"""

import random
import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

SEED=10
random.seed(SEED)
torch.manual_seed(SEED)

class Respiratory_Deteroration(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.d1 = layers.Dense(308, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.25)        
        self.d2 = layers.Dense(231, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.d3 = layers.Dense(1, activation='sigmoid')

    def forward(self, x,train=False):
        x = self.d1(x)
        x = self.dropout1(x, training=train)
        x = self.d2(x)
        x = self.dropout2(x, training=train)
        x = self.d3(x)     
        return x

    def call(self, inputs):
        x = self.forward(inputs, train=False)
        return x 

class PatientDeteriorationDataset(Dataset):
    def __init__(self, data, used_features, target_label):
        self.labels = data[target_label].to_numpy(dtype=np.int64)
        self.features = data[used_features]
        self.features = self.features.drop(columns=['label']).to_numpy(dtype=np.float32)
        ind = np.where(self.labels>0)[0]
        self.labels[ind] = 1
           
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_predictions(model,loader):
    pred_prob = []
    true_labels = []

    for x,y in loader:
        x = x.numpy()
        y = y.numpy()
        
        pred_part = model.forward(x, train=False)[:,0]
        pred_prob.append(pred_part)
        
        ind = np.where(y>0)[0]
        y[ind] = 1
        
        true_labels.append(y)
        
    return np.concatenate(pred_prob), np.concatenate(true_labels)
        
def normal_model_training(model,train_loader,val_loader,loss_fn,opt,name,epochs=50):
    best=0
    TL=[]
    VL=[]
    Val_auc=[]
    BE=0

    for epoch in range(0, epochs):
        loss = 0

        for i, data in enumerate(train_loader):
            inputs, targets = data
            
            inputs = inputs.numpy()
            targets = targets.numpy()
            
            ind = np.where(targets>0)[0]
            targets[ind] = 1
            
            with tf.GradientTape() as tape1:
                 p = model.forward(inputs, train=True)[:,0]
                 l = loss_fn(p, targets)
            
            grad = tape1.gradient(l,model.trainable_variables)
            opt.apply_gradients(zip(grad,model.trainable_variables))
            loss = loss+l
        
        TL.append(loss/(i+1)) 
        Preds, Labels = get_predictions(model, val_loader)
        
        val_auc = roc_auc_score(Labels, Preds)
        
        VL.append(loss_fn(Preds,Labels))
        Val_auc.append(val_auc)
        
        if best < val_auc:
           best = val_auc
           model.save_weights(f'./models/{name}.h5')
           BE = epoch
           
        print(f"Epoch: {epoch:.1f} Train Loss: {TL[-1]:.5f} Val Loss: {VL[-1]:.5f} Test: {Val_auc[-1]:.5f}")   
        
        if (epoch-BE) > 25:
           break 

    model.load_weights(f'./models/{name}.h5')

    return TL, VL, Val_auc, model    

            
all_no_baseline = ['age', 'sex', 'Vital_Signs HR', 'Vital_Signs RR', 'Vital_Signs SBP', 'Vital_Signs TEMP',
                   'Vital_Signs SPO2', 'Vital_Signs FiO2', 'Vital_Signs masktyp', 'Vital_Signs avpu',
                   'Blood_Test ALT-IU/L', 'Blood_Test CRP-mg/L', 'Blood_Test Albumin-g/L', 'Blood_Test Urea-mmol/L',
                   'Blood_Test Sodium-mmol/L', 'Blood_Test Haematocrit-L/L', 'Blood_Test Haemoglobin-g/dL',
                   'Blood_Test Bilirubin-umol/L', 'Blood_Test Potassium-mmol/L', 'Blood_Test Basophils-x10^9/L',
                   'Blood_Test Creatinine-umol/L', 'Blood_Test MeanCellVol-fL', 'Blood_Test Monocytes-x10^9/L',
                   'Blood_Test Platelets-x10^9/L', 'Blood_Test Eosinophils-x10^9/L', 'Blood_Test Lymphocytes-x10^9/L',
                   'Blood_Test Neutrophils-x10^9/L', 'Blood_Test WhiteCells-x10^9/L', 'Blood_Test Alk.Phosphatase-IU/L',
                   'Blood_Gas BE ACT (BG)', 'Blood_Gas BE STD (BG)', 'Blood_Gas BICARB (BG)', 'Blood_Gas CA+ + (BG)',
                   'Blood_Gas CL- (BG)', 'Blood_Gas CLAC (BG)', 'Blood_Gas CREAT (BG)', 'Blood_Gas CTO2C (BG)',
                   'Blood_Gas ESTIMATED OSMOLALITY (BG)', 'Blood_Gas FCOHB (BG)', 'Blood_Gas FHHB (BG))',
                   'Blood_Gas FIO2', 'Blood_Gas GLUCOSE (BG)', 'Blood_Gas HB (BG)', 'Blood_Gas HCT (BG)',
                   'Blood_Gas K+ (BG)', 'Blood_Gas METHB (BG)', 'Blood_Gas NA+ (BG)', 'Blood_Gas O2 SAT (BG)',
                   'Blood_Gas P5OC (BG)', 'Blood_Gas PCO2 POC', 'Blood_Gas PH (BG)', 'Blood_Gas PO2 (BG)',
                   'Blood_Gas TEMPERATURE POCT', 'Var_Mean_Vital_Signs HR', 'Max_Min_Vital_Signs HR',
                   'Delta_Mean_Vital_Signs HR', 'Var_Mean_Vital_Signs RR', 'Max_Min_Vital_Signs RR',
                   'Delta_Mean_Vital_Signs RR', 'Var_Mean_Vital_Signs SBP', 'Max_Min_Vital_Signs SBP',
                   'Delta_Mean_Vital_Signs SBP', 'Var_Mean_Vital_Signs TEMP', 'Max_Min_Vital_Signs TEMP',
                   'Delta_Mean_Vital_Signs TEMP', 'Var_Mean_Vital_Signs SPO2', 'Max_Min_Vital_Signs SPO2',
                   'Delta_Mean_Vital_Signs SPO2', 'Var_Mean_Vital_Signs FiO2', 'Max_Min_Vital_Signs FiO2',
                   'Delta_Mean_Vital_Signs FiO2', 'Var_Mean_Vital_Signs masktyp', 'Max_Min_Vital_Signs masktyp',
                   'Delta_Mean_Vital_Signs masktyp', 'Var_Mean_Vital_Signs avpu', 'Max_Min_Vital_Signs avpu',
                   'Delta_Mean_Vital_Signs avpu']


def get_names_of_predefined_feature_sets():
    return all_no_baseline            
