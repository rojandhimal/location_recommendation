from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import UserRegisterForm
from django.contrib.auth import get_user_model
from .models import LocationData, UserProfile, LocationRated
from django.core.paginator import Paginator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.core.cache import cache
import numpy as np
import pandas as pd
from ast import literal_eval
import copy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

tknzr = WordPunctTokenizer()
nltk.download('stopwords')
stoplist = stopwords.words('english')
from nltk.stem.porter import PorterStemmer

from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

# umatrixpath = '/mnt/c/Users/psychogyio/Desktop/andrea/machine_learning_for_the_web-master_3/chapter_7/server_locationrecsys/umatrix.csv'

nlocationsperquery = 10
nminimumrates = 5
numrecs = 5
recmethod = 'cf_itembased'

User = get_user_model()


def PreprocessTfidf(texts, stoplist=[], stem=False):
    newtexts = []
    for text in texts:
        if stem:
            tmp = [w for w in tknzr.tokenize(text) if w not in stoplist]
        else:
            tmp = [stemmer.stem(w) for w in [w for w in tknzr.tokenize(text) if w not in stoplist]]
        newtexts.append(' '.join(tmp))
    return newtexts


def index(request):
    return render(request, 'home.html')


def SignUpUser(request):
    form = UserRegisterForm()

    if request.method == 'GET':
        context = {'form': form}
        return render(request, 'signup.html', context)

    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            uprofile = UserProfile()
            uprofile.user = user
            uprofile.name = user.username
            uprofile.save(create=True)
            username = form.cleaned_data['username']
            messages.success(request, f'Registered successfully as {username}')
            return redirect("recommendationsystem:login")
        else:
            for msg in form.errors:
                messages.error(request, form.errors[msg].as_text())
                return redirect("recommendationsystem:register")


def LoginUser(request):
    if request.method == 'POST':
        print(request.POST)
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.info(request, f"You are now logged in as {username}")
            return redirect('recommendationsystem:search')
        else:
            messages.error(request, "Invalid username or password.")
            return redirect('recommendationsystem:login')
    return render(request, "login.html")


def LogoutUser(request):
    logout(request)
    messages.success(request, 'Logged out successfully')
    return redirect('recommendationsystem:login')


def search(request):
    # if not request.user.is_authenticated:
    #     messages.error(request, 'You must be logged in to view this page.')
    #     return redirect('recommendationsystem:login')
    if request.method == 'GET':
        data = request.GET.get('q', None)
        if data is None:
            # context = {
            #     'msg': 'Please make sure you rate 5 locations from the search list.'
            # }
            return render(request, 'search.html')
        else:
            print(data)
            titles = cache.get('titles')
            if titles is None:
                texts = []
                locations = LocationData.objects.all()
                ndim = locations[0].ndim
                matr = np.empty([1, ndim])
                titles_list = []
                cnt = 0
                for loc in locations[:]:
                    texts.append(loc.description)
                    newrow = np.array(loc.array)
                    if cnt == 0:
                        matr[0] = newrow
                    else:
                        matr = np.vstack([matr, newrow])
                    titles_list.append(loc.title)
                    cnt += 1
                vectorizer = TfidfVectorizer(min_df=1, max_features=True)
                processedtexts = PreprocessTfidf(texts, stoplist, True)
                model = vectorizer.fit(processedtexts)
                cache.set('model', model)
                cache.set('data', matr)
                cache.set('titles', titles_list)
            else:
                print('loaded', str(len(titles)))

            Umatrix = cache.get('umatrix')
            if Umatrix.any() == None:
                df_umatrix = pd.read_csv(umatrixpath)
                Umatrix = df_umatrix.values[:, 1:]
                print('umatrix:', Umatrix.shape)
                cache.set('umatrix', Umatrix)
                cf_itembased = CF_itembased(Umatrix)
                cache.set('cf_itembased', cf_itembased)

            # load all locations vectors/titles
            matr = cache.get('data')
            print('matr', len(matr))
            titles = cache.get('titles')
            print('ntitles:', len(titles))
            model_tfidf = cache.get('model')

            # load in cache rec sys methods
            print('load methods...')

            # find locations similar to the query
            # print 'names:',len(model_tfidf.get_feature_names())
            queryvec = model_tfidf.transform([data.lower().encode('ascii', 'ignore')]).toarray()

            # print 'vec:', queryvec

            sims = cosine_similarity(queryvec, matr)[0]
            indxs_sims = list(sims.argsort()[::-1])
            # print indxs_sims
            titles_query = list(np.array(titles)[indxs_sims][:nlocationsperquery])
            if not titles_query:
                found = False
            else:
                found = True
                locations = list(zip(titles_query, indxs_sims[:nlocationsperquery]))
                locations_li = []
                for loc, indx in locations:
                    location_obj = LocationData.objects.get(title=loc)
                    locations_li.append(location_obj)
            show_rec_btn = False
            if request.user.is_authenticated:
                user_profile = UserProfile.objects.get(user=request.user)
                rated_locations = user_profile.ratedlocations.all()
                msg = f"Rated Locations: {len(rated_locations)}"
                if len(rated_locations) < nminimumrates:
                    min_rated_msg = f"{nminimumrates - len(rated_locations)} more locations left to be rated."
                elif len(rated_locations) >= nminimumrates:
                    min_rated_msg = "Minimum criteria satisfied !!"
                    show_rec_btn = True

            context = {
                'locations': locations,
                'rates': [1, 2, 3, 4, 5],
                'found': found,
                'locations_list': locations_li,
                'msg': msg,
                'min_rated_msg': min_rated_msg,
                'show_rec_btn': show_rec_btn
            }

            return render(
                request, 'search.html', context)


def RemoveFromList(liststrings, string):
    outlist = []
    for s in liststrings:
        if s == string:
            continue
        outlist.append(s)
    return outlist


def rate_location(request):
    if request.user.is_authenticated:
        rate = request.GET.get("vote")
        locations = request.GET.get("locations")
        location = request.GET.get("location")
        location_indx = request.GET.get("locationindx")
        locations, locationsindxs = list(zip(*literal_eval(locations)))
        locationindx = int(location_indx)

        # save location rating

        userprofile = UserProfile.objects.get(user=request.user)
        location_obj = LocationData.objects.get(title=location)

        if LocationRated.objects.filter(location=location_obj).filter(user=userprofile).exists():
            lr = LocationRated.objects.get(location=location_obj, user=userprofile)
            lr.value = int(rate)
            lr.save()
        else:
            lr = LocationRated()
            lr.user = userprofile
            lr.value = int(rate)
            lr.location = location_obj
            lr.locationindx = locationindx
            lr.save()

        userprofile.save()
        rated_locations = userprofile.ratedlocations.all()
        msg = f"Locations Rated: {len(rated_locations)}"
        show_rec_btn = False
        if len(rated_locations) < nminimumrates:
            min_rated_msg = f"{nminimumrates - len(rated_locations)} more locations left to be rated."
        elif len(rated_locations) >= nminimumrates:
            min_rated_msg = "Minimum criteria satisfied !!"
            show_rec_btn = True

        locations = RemoveFromList(locations, location)
        # print(locations)
        locationsindxs = RemoveFromList(locationsindxs, locationindx)
        locations_li = []
        for loc in locations:
            location_obj = LocationData.objects.get(title=loc)
            locations_li.append(location_obj)
        context = {"locations": list(zip(locations, locationsindxs)),
                   "found": True,
                   "rates": [1, 2, 3, 4, 5],
                   "locations_list": locations_li,
                   "msg": msg,
                   "min_rated_msg": min_rated_msg,
                   "show_rec_btn": show_rec_btn}
        return render(
            request, 'search.html', context)
    else:
        messages.info(request, "You must be logged in to rate a location.")
        return redirect('recommendationsystem:search')


def location_recs(request):
    userprofile = None
    if request.user.is_authenticated:
        userprofile = UserProfile.objects.get(user=request.user)
    else:
        messages.info(request, 'You must be logged in to view recommendations.')
        return redirect('recommendationsystem:login')

    ratedlocations = userprofile.ratedlocations.all()
    # alllocation = Locations.objects.all()
    print('rated:', ratedlocations, '--', [l.locationindx for l in ratedlocations])
    
    context = {}
    if len(ratedlocations) < nminimumrates:
        context['underminimum'] = True
        context['nrates'] = len(ratedlocations)
        context['nminimumrates'] = nminimumrates
        return render(
            request, 'recommendations.html', context)

    u_vec = literal_eval(userprofile.array)
    print("This is u vec",u_vec)
    print(len(u_vec))
    # print 'uu:',u_vec
    Umatrix = cache.get('umatrix')
    print(Umatrix)
    # print Umatrix.shape,'--',len(u_vec)
    locationslist = cache.get('titles')
    # recommendation...
    u_rec = None

    # cf_itembased
    cf_itembased = cache.get('cf_itembased')
    if cf_itembased is None:
        cf_itembased = CF_itembased(Umatrix)
    u_rec = cf_itembased.CalcRatings(u_vec, numrecs)

    # save last recs
    userprofile.save(recsvec=u_rec)
    # print(np.array(locationslist))
    # print(u_rec)
    # print(numrecs)
    # context['recs'] = list(np.array(locationslist)[list(u_rec)][:numrecs])
    # print(list(u_vec))
    # print(locationslist)
    rec_indxs = [idx for idx, val in enumerate(u_vec) if val != 0]
    # print(list(np.array(locationslist)[rec_indxs]))
    location_objs = []
    for loc in locationslist:
        location_obj = LocationData.objects.get(title=loc)
        location_objs.append(location_obj)

    # print(location_ratings)
    # print(list(np.array(locationslist)[:numrecs]))
    context['recs'] = list(np.array(locationslist)[rec_indxs])
    context['recommendations'] = location_objs
    # context['location_ratings'] = location_ratings
    return render(
        request, 'recommendations.html', context)


def about(request):
    return render(request, 'about.html')


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
        items = np.argsort(self.simmatrix[r])[::-1]
        items = items[items != r]
        cnt = 0
        neighitems = []
        for i in items:
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

    def CalcRatings(self, u_vec, K):
        u_rec = copy.copy(u_vec)
        for r in range(len(u_vec)):
            if u_vec[r] == 0:
                neighitems = self.GetKSimItemsperUser(r, K, u_vec)
                # calc predicted rating
                u_rec[r] = self.CalcRating(r, u_vec, neighitems)
        # take out the rated locations
        seenindxs = [indx for indx in range(len(u_vec)) if u_vec[indx] > 0]
        u_rec[seenindxs] = -1
        recsvec = np.argsort(u_rec)[::-1][np.argsort(u_rec) > 0]

        return recsvec






from recommendationsystem.models import LocationData,LocationRated
from sklearn.metrics.pairwise import cosine_similarity


def New_Rec(request):
    userprofile = None
    if request.user.is_authenticated:
        userprofile = UserProfile.objects.get(user=request.user)
    else:
        messages.info(request, 'You must be logged in to view recommendations.')
        return redirect('recommendationsystem:login')
    ratedlocations = userprofile.ratedlocations.all()
    print('rated:', ratedlocations, '--', [l.locationindx for l in ratedlocations])
    context = {}
    if len(ratedlocations) < nminimumrates:
        context['underminimum'] = True
        context['nrates'] = len(ratedlocations)
        context['nminimumrates'] = nminimumrates
        return render(
            request, 'recommendations.html', context)

    # Read all data from database
    # all location data
    all_location = LocationData.objects.all() 
    print("all loc\n")
    print(all_location)
    # all rating in location data
    rated_location = LocationRated.objects.all()
    # all ratings made by user
    user_rated_locations = userprofile.ratedlocations.all()

    # converting data to dataframe and filtering data
    location_data_df = pd.DataFrame(list(LocationData.objects.all().values()))
    rated_location_df = pd.DataFrame(list(LocationRated.objects.all().values()))
    user_rated_locations = pd.DataFrame(list(userprofile.ratedlocations.all().values()))
    print(user_rated_locations)
    user_avg_rating = user_rated_locations['value'].mean()

    # get only id and title
    location_data_df=location_data_df[['id','title']]
    # get user id locatio id and vale i.e rating
    rated_location_df = rated_location_df[['user_id','location_id','value']]
    rated_location_df['id']=rated_location_df['location_id']
    # get title of the location that is rated by user that is logged in
    user_rated_locations = user_rated_locations[['user_id','location_id']]
    print("User rated location mean =>",user_avg_rating)


    print("this is all rated  data")
    print(rated_location_df) 
    print("\m #########")
    all_location = location_data_df.join(rated_location_df,on='id',how='left', lsuffix='_left', rsuffix='_right')
    all_location = all_location.fillna(0)
    print("all_location after nan 0")
    print(all_location)
    print("##########$")

    # this gives average ratings from data that is associated to location name
    ratings = pd.DataFrame(all_location.groupby('location_id')['value'].mean())
    print("This is averege rating \n")
    print(ratings)
    all_rated_loc = location_data_df.join(ratings,on='id',how='left', lsuffix='_left', rsuffix='_right')
    all_rated_loc = all_rated_loc.fillna(0)
    print("pivot average ##\n")
    print(all_rated_loc)
    print("#####")

    rec_matrix_UII = all_rated_loc.pivot_table(index='id', columns='title', values='value')
    print("$$$$$$$$$$")
    rec_matrix_UII = rec_matrix_UII.fillna(0)
    print("#$%#$@")
    recommender  = cosine_similarity(rec_matrix_UII)
    print("no res after this")
    recommender_df = pd.DataFrame(recommender, 
                                  columns=rec_matrix_UII.index,
                                  index=rec_matrix_UII.index)
    print(recommender_df)
    ## Item Rating Based Cosine Similarity
    print("last")
    
    recs = recommender_df[user_rated_locations['location_id']]
        
    user_rec = recs.mean(axis=1).sort_values(ascending=False)
    top_five_rec_id =user_rec.head(5)
    typ_arr = top_five_rec_id.index.array
    print(typ_arr)
    rec_title = location_data_df.loc[location_data_df['id'].isin(typ_arr)]
    print("REc title",rec_title)
    print("This is recommendation id=>",top_five_rec_id)
    # rec_location = LocationData.objects.filter(id=top_five_rec_id)
    # print(rec_location)   
    obj = []
    
    for title in rec_title['title']:
        l =LocationData.objects.get(title=title)
        obj.append(l)
       
    context = {"locations":obj}
    return render(
        request, 'recommendations.html', context)

    


   