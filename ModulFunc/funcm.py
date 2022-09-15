import re
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

class functions:
    def load_data(self,file:str):
        """
        Loading a excel file
        :param file: File Target
        """

        data = pd.read_excel(f'{file}')
        data = data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
        return data


    def remove_username(self,dataframe, column: str):
        """
        Removing username in dataframe.You should this before use cleaning_data function
        :param dataframe: Dataframe Target
        :param column: Column Target
        """
        final = []
        for text in dataframe[column].values.astype(str):
            text = re.sub(r'@[\w]+', '', text)
            final.append(text)
            data = pd.DataFrame(final, columns=[column])
        return data


    def cleaning_data(self,dataframe, column: str, hastag_return: bool):
        """
        Removing all character,symbol and etc in dataframe
        :param dataframe: Dataframe Target
        :param column: Column Target
        :param hastag_return: Return Hastag
        """
        hastag = dataframe[column].apply(lambda x: re.findall(r'\B#\w*[a-zA-Z]+\w*', x))
        text = dataframe[column].str.replace(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', regex=True)
        text = text.str.replace(r'#(\w+)', '', regex=True)
        text = text.str.replace(r"[\"\'\|\?\=\.\\*\+\!\:\,]", '', regex=True)
        text = text.str.replace(r'\d+', '', regex=True)
        text = text.str.replace(r'RT', '', regex=True)
        text = text.str.replace(r'\n', ' ', regex=True)
        text = text.str.replace(r'\s\s+', " ", regex=True)
        text = text.str.lstrip()
        text = text.str.lower()
        if hastag_return == True:
            hastag = hastag.to_frame(name="hastag")
            text = text.to_frame(name=column)
            return pd.concat([text, hastag], axis=1)
        else:
            return text.to_frame(name=column)


    def remove_stopwords(self,dataframe, column: str, lang: str):
        """
        Removing stopwords inside dataframe. Need install NLTK Stopwords to specify stopword list
        :param dataframe: Dataframe Target
        :param column: Column Target
        :param lang: Language
        """

        stop = stopwords.words({lang})
        data = dataframe[column].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
        return data.to_frame(name=column)


    def write_json(self,filename:str,dataframe):
        """
        Writing a json file
        :param filename: Ouput Name
        :param dataframe: Dataframe Target
        :return:
        """
        with open(f'{filename}.json', 'w') as f:
            f.write(dataframe.to_json(orient="records", lines=False))


    def printsome(self,word:str):
        print(word)