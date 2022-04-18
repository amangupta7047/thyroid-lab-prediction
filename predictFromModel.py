import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
import numpy as np

import pickle


class prediction:

    def __init__(self):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        

    def predictionFromModel(self):

        try:
            
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()

            

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            data = preprocessor.dropUnnecessaryColumns(data,
                                                       ['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured',
                                                        'FTI_measured', 'TBG_measured', 'TBG', 'referral_source'])

            data = preprocessor.replaceInvalidValuesWithNull(data)

            data = preprocessor.encodeCategoricalValuesPrediction(data)
            is_null_present=preprocessor.is_null_present(data)
            if(is_null_present):
                data=preprocessor.impute_missing_values(data)


            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
            kmeans=file_loader.load_model('KMeans')

            clusters=kmeans.predict(data)#drops the first column for cluster prediction
            data['clusters']=clusters
            clusters=data['clusters'].unique()
            result=[] # initialize balnk list for storing predicitons
            with open('EncoderPickle/enc.pickle', 'rb') as file: #let's load the encoder pickle file to decode the values
                encoder = pickle.load(file)

            for ind in data.index:
                # cluster_data=i
                # print(type(cluster_data))
                # print(cluster_data)
                cluster_number=int(data['clusters'][ind])
                # print(cluster_number)

                query = [data['age'][ind],data['sex'][ind],data['on_thyroxine'][ind],data['query_on_thyroxine'][ind],data['on_antithyroid_medication'][ind],data['sick'][ind],
                                    data['pregnant'][ind],data['thyroid_surgery'][ind],data['I131_treatment'][ind],data['query_hypothyroid'][ind],data['query_hyperthyroid'][ind],
                                    data['lithium'][ind],data['goitre'][ind],data['tumor'][ind],data['hypopituitary'][ind],data['psych'][ind],
                                    data['TSH'][ind],data['T3'][ind],data['TT4'][ind],data['T4U'][ind],data['FTI'][ind]]
                # print(query)    
            
                query = np.array(query).reshape(1,21)

                model_name = file_loader.find_correct_model_file(cluster_number)
                model = file_loader.load_model(model_name)
                for val in (encoder.inverse_transform(model.predict(query))):
                    result.append(val)
            result = pandas.DataFrame(result,columns=['Predictions'])
            path="Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv",header=True) #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
            
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path

            