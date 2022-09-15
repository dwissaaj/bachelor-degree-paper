import re
import warnings
import gensim
import gensim.models.ldamodel
import matplotlib.pyplot as plt
import pyLDAvis
from gensim import corpora
from gensim.models import TfidfModel
from nltk.tokenize import word_tokenize
from pyLDAvis import gensim_models
from wordcloud import WordCloud
from Unsupervised.funcm import functions
warnings.filterwarnings("ignore", category=DeprecationWarning)
fc = functions()
class Lda():
    def generate_word(dataframe, column: str):
        """
        :param dataframe: Dataframe
        :param column: Target Column
        """
        data = dataframe[column].apply(word_tokenize)
        return data


    def id_to_word(dataframe):
        """
        :param dataframe: Dataframe Target
        """
        idto_word = corpora.Dictionary(dataframe)
        return idto_word


    def create_corpus(dataframe, column: str):
        """
        :param dataframe: Dataframe
        :param column: Target Column
        """

        generated = Lda.generate_word(dataframe, f"{column}")
        idtwords = corpora.Dictionary(generated)
        cps = []
        for text in generated:
            new = idtwords.doc2bow(text)
            cps.append(new)
        return cps


    def lda_model(corpuslda, id_two_w, topic: int):
        """
        :param corpuslda: Corpus Result
        :param id_two_w: id2word Result
        :param topic: Number of Topic
        """
        lda_mdl = gensim.models.ldamodel.LdaModel(corpus=corpuslda,
                                                  id2word=id_two_w,
                                                  num_topics=topic
                                                 )
        return lda_mdl


    def visualize(lda_data, corpus_list, id2word_data, mds: str, r: int,filename:str):
        '''
        :param lda_data: LDA Model Result
        :param corpus_list: Corpus Result
        :param id2word_data: id2word Result
        :param mds: Default is "mmds"
        :param r: Int Value
        :param filename: File Name Output
        :return:

        '''
        vis = pyLDAvis.gensim_models.prepare(topic_model=lda_data,
                                             corpus=corpus_list,
                                             dictionary=id2word_data,
                                             mds=mds,
                                             R=r)
        pyLDAvis.save_html(vis,f"{filename}.html")
        return vis


    def remove_high_tfidf(dataframe, column: str, low_value: float):
        """
        USE THIS IF YOU HAVE A HIGH TFIDF WORD
        USE IT CAREFULLY FOR BACK UP CHECK MANUAL BOOK AT TXT
        THIS FUNCTION NEED CORPUS AND TFIDF
        :param dataframe: DATAFRAME YOU ARE USING
        :param column: COLUMN NAME
        :param low_value: MINIMUM OF VALUE
        :return:
        """
        generated_w = Lda.generate_word(dataframe, f"{column}")
        id2w = Lda.id_to_word(generated_w)
        corpustfidf = [id2w.doc2bow(text) for text in generated_w]
        tfidf = TfidfModel(corpustfidf, id2word=id2w)
        low_value = low_value
        words = []
        words_missing_in_tfidf = []

        for i in range(0, len(corpustfidf)):
            bow = corpustfidf[i]
            low_value_words = []  # reinitialize to be safe. You can skip this.
            tfidf_ids = [id for id, value in tfidf[bow]]
            bow_ids = [id for id, value in bow]
            low_value_words = [id for id, value in tfidf[bow] if value < low_value]
            drops = low_value_words + words_missing_in_tfidf
            for item in drops:
                words.append(id2w[item])
            words_missing_in_tfidf = [id for id in bow_ids if
                                      id not in tfidf_ids]  # The words with tf-idf socre 0 will be missing

            new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
            corpustfidf[i] = new_bow
        return corpustfidf, id2w



    def get_hastag(dataframe,column:str)-> list:
        """

        :param dataframe: Dataframe
        :param column: Target Column
        :return: List of Hastag
        """

        nan_value = float("NaN")
        dataframe[column] = dataframe[column].apply(lambda x: re.findall(r'\B#\w*[a-zA-Z]+\w*', x))
        hastag = dataframe.astype(str).drop_duplicates()
        hastag = hastag.replace({"\[": "",
                              "\]": "",
                              "\'": "",
                              "\,": " "}, regex=True)
        hastag.replace("", nan_value, inplace=True)
        hastag.dropna(subset = [column], inplace=True)
        tagstr = hastag[column].values.tolist()
        return tagstr

    def wordcloud_maker(stack:list,colormap:str):
        '''

        :param stack: The list of the hastag
        :param colormap: color map in matplotlib
        :return
        '''
        to_str = ' '.join(map(str,stack))
        wordcloud = WordCloud(width=2000, height=2000,max_words=75, colormap=f"{colormap}").generate(to_str)
        plt.figure()
        plt.imshow(wordcloud, interpolation="lanczos")
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.show()


"""
df = load_data("Voxpopdata.xlsx")
remove_user = remove_username(df,"text")
clean_data = cleaning_data(remove_user,"text",False)
remove_stop = remove_stopwords(clean_data,"text","indonesian")
toenize = generate_word(clean_data,"text")
id2word = id_to_word(toenize)
corpus = create_corpus(remove_stop,"text")
vish = visualize(lda,corpus,id2word,"mmds",10,"normal")

ldah = lda_model(corpus,id2word,25 )
hastag = get_hastag(df,"text")
wordcloud_maker(hastag,"Set2")
"""

""""
lowcor = remove_high_tfidf(clean_data,"text",0.1)[0]
lowid = remove_high_tfidf(clean_data,"text",0.1)[1]
ldal = lda_model(lowcor,lowid,25 )

"""



