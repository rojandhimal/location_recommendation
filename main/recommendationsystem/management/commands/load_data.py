from django.core.management.base import BaseCommand
import os
import optparse
import numpy as np
import pandas as pd
import math
import json
import copy
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

tknzr = WordPunctTokenizer()
nltk.download('stopwords')
stoplist = stopwords.words('english')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
from recommendationsystem.models import LocationData
from django.core.cache import cache


# python manage.py load_data --input=locations1.csv --nmaxwords=30000  --umatrixfile=matrix-final.csv
class Command(BaseCommand):
    """
    option_list = BaseCommand.option_list + (
            optparse.make_option('-i', '--input', dest='input',
                                 type='string', action='store',
                                 help=('Input plots file')),
            optparse.make_option('--nmaxwords', '--nmaxwords', dest='nmaxwords',
                                 type='int', action='store',
                                 help=('nmaxwords')),
            optparse.make_option('--umatrixfile', '--umatrixfile', dest='umatrixfile',
                                 type='string', action='store',
                                 help=('umatrixfile')),
        )
    """

    def add_arguments(self, parser):

        parser.add_argument(
            '--input',
            action='store',
            help='Input plots file')

        parser.add_argument(
            '--nmaxwords',
            action='store',
            type=int,
            help='nmaxwords')

        parser.add_argument(
            '--umatrixfile',
            action='store',
            help='umatrixfile')

    def PreprocessTfidf(self, texts, stoplist=[], stem=False):
        newtexts = []
        for i in range(len(texts)):
            text = texts[i]
            if stem:
                tmp = [w for w in tknzr.tokenize(text) if w not in stoplist]
            else:
                tmp = [stemmer.stem(w) for w in [w for w in tknzr.tokenize(text) if w not in stoplist]]
            newtexts.append(' '.join(tmp))
        return newtexts

    def handle(self, *args, **options):
        input_file = options['input']

        df = pd.read_csv(input_file)
        print(df.head(2))
        tot_textplots = df['plot'].tolist()
        tot_titles = df['title'].tolist()
        tot_pictures = df['picture'].tolist()
        tot_addresses = df['address'].tolist()
        nmaxwords = options['nmaxwords']
        vectorizer = TfidfVectorizer(min_df=0, max_features=nmaxwords)
        processed_plots = self.PreprocessTfidf(tot_textplots, stoplist, True)
        mod_tfidf = vectorizer.fit(processed_plots)
        vec_tfidf = mod_tfidf.transform(processed_plots)
        ndims = len(mod_tfidf.get_feature_names())
        nlocations = len(tot_titles[:])

        # delete all data
        LocationData.objects.all().delete()

        matr = np.empty([1, ndims])
        print(ndims)
        titles = []
        cnt = 0
        for l in range(nlocations):
            locationdata = LocationData()
            locationdata.title = tot_titles[l]
            locationdata.image = tot_pictures[l]
            locationdata.address = tot_addresses[l]
            locationdata.description = tot_textplots[l]
            locationdata.ndim = ndims
            locationdata.array = json.dumps(vec_tfidf[l].toarray()[0].tolist())
            locationdata.save()
            newrow = json.loads(locationdata.array)
            if cnt == 0:
                matr[0] = newrow
            else:
                matr = np.vstack([matr, newrow])
            titles.append(locationdata.title)
            cnt += 1
        # cached
        cache.set('data', matr)
        cache.set('titles', titles)
        titles = cache.get('titles')
        # print titles
        print('len:', len(titles))
        cache.set('model', mod_tfidf)

        # load the utility matrix
        umatrixfile = options['umatrixfile']
        df_umatrix = pd.read_csv(umatrixfile)
        print(df_umatrix.head(2))

        Umatrix = df_umatrix.values[:, 1:]
        print('umatrix:', Umatrix.shape)
        cache.set('umatrix', Umatrix)
        # load rec methods...
        cf_itembased = CF_itembased(Umatrix)
        cache.set('cf_itembased', cf_itembased)

        # test...
        model_vec = cache.get('model')
        # print 'mod:',model_vec,'--',mod_tfidf
        print('nwords:', len(model_vec.get_feature_names()))
        # print 'vec:',model_vec.transform(['wars star'])


from scipy.stats import pearsonr
from scipy.spatial.distance import cosine


def sim(x, y, metric='cos'):
    if metric == 'cos':
        return 1. - cosine(x, y)
    else:  # correlation
        return pearsonr(x, y)[0]


class CF_itembased(object):
    def __init__(self, data):
        # calc item similarities matrix
        nitems = len(data[0])
        self.data = data
        self.simmatrix = np.zeros((nitems, nitems))
        for i in range(nitems):
            for j in range(nitems):
                if j >= i:  # triangular matrix
                    self.simmatrix[i, j] = sim(data[:, i], data[:, j])
                else:
                    self.simmatrix[i, j] = self.simmatrix[j, i]

    def GetKSimItemsperUser(self, r, K, u_vec):
        # print(r)
        # print(K)
        # print(u_vec)
        # print(self.simmatrix[r])
        items = np.argsort(self.simmatrix[r])[::-1]
        items = items[items != r]
        cnt = 0
        neighitems = []
        print(u_vec)
        for i in items:
            print(i)
            print(u_vec[i])
            if u_vec[i] > 0 and cnt < K:
                neighitems.append(i)
                cnt += 1
            elif cnt == K:
                break
        return neighitems

    def CalcRating(self, r, u_vec, neighitems):
        rating = 0.
        den = 0.
        for i in neighitems:
            rating += self.simmatrix[r, i] * u_vec[i]
            den += abs(self.simmatrix[r, i])
        if den > 0:
            rating = np.round(rating / den, 0)
        else:
            rating = np.round(self.data[:, r][self.data[:, r] > 0].mean(), 0)
        return rating

    def CalcRatings(self, u_vec, K, indxs=False):
        u_rec = copy.copy(u_vec)
        for r in range(len(u_vec) - 1): #edited and added -1
            if u_vec[r] == 0:
                neighitems = self.GetKSimItemsperUser(r, K, u_vec)
                # calc predicted rating
                u_rec[r] = self.CalcRating(r, u_vec, neighitems)
        if indxs:
            # take out the rated locations
            seenindxs = [indx for indx in range(len(u_vec)) if u_vec[indx] > 0]
            u_rec[seenindxs] = -1
            recsvec = np.argsort(u_rec)[::-1][np.argsort(u_rec) > 0]

            return recsvec
        return u_rec



