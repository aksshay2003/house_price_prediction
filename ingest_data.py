import pandas as pd
import os
import zipfile
from abc import ABC,abstractmethod #abc-abstract base class

class DataIngestor(ABC): #ABC uses to define abstract class
    @abstractmethod
    def ingest(self, filename:str) -> pd.DataFrame:
        pass
    
# class for ingesting zip file
#what most of the people do di write read data function directly instead for readability we write ingest function different data reads(json,scv..etc)
class zipDataIngestor(DataIngestor): # extends DataIngestor
    def ingest(self,filename:str)-> pd.DataFrame:
        if not filename.endswith(".zip"):
            raise ValueError("file is not in zip format")
        with zipfile.ZipFile(filename,"r") as zip_file:
            # extract file into extracted_data
            zip_file.extractall("extracted_data")
            #extract csv file from it
            extracted_files=os.listdir("extracted_data")
            csv_files=[files for files in extracted_files if files.endswith(".csv")]
            if(len(csv_files)==0):
                raise ValueError("no csv file is found")
            csv_file_path=os.path.join("extracted_data",csv_files[0])
            df=pd.read_csv(csv_file_path)
            return df
class DataIngestorFactory:
    @staticmethod
    # @staticmethod: This method can be called without creating an instance of DataIngestorFactory.

    def get_data_ingester(file_ext_name:str)-> DataIngestor: # ->DataIngestor is a return type annotation meaing return an instance(or subclass of it) of it..it improves readability
        # static methods doestnt need self
        if(file_ext_name.endswith(".zip")):
            return zipDataIngestor()
        else:
            raise ValueError("not a zip file")

if __name__=="__main__":
    filepath="Data\\Dataset.zip"
    #file extension
    file_ext=os.path.splitext(filepath)[1]
    data_ingester=DataIngestorFactory.get_data_ingester(file_ext)
    df=data_ingester.ingest(filepath)
    print(df.head())