# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from rest_framework.views import APIView
from rest_framework.response import Response
#for function based views
from rest_framework.decorators import api_view
#for apiroot reverse
from rest_framework.reverse import reverse


import pandas as pd
from api import model_fr
from api import vectTrain_fr
from api import model_arabizi
from api import vectTrain_arabizi
from api import model_arabic
from api import vectTrain_arabic




def api_root(request):
    """
    The root of all APIs, serves as a basic presentation of the APIs aviliable,
    however needs manual additions of the functions.
    reverse() serves as a url call to each function views.
    """
    return Response({
        'hello_world': reverse('hello_world', request=request),
        'add': reverse('sentiment', request=request),
    })

@api_view()
def hello_world(request):
    """
    An example api, this part of text will be visible when entering /hello_world.
    """
    return Response({"message": "Hello, world!"})

@api_view()
def sentiment(request):
    try:
        comment = request.GET.get('comment')
        lan = request.GET.get('lan')
        comment.replace("%20"," ")
        comment_class = "Neutral"


        df_t = pd.DataFrame(columns=["Comment"])
        df_t.loc[1] = comment
        if lan=="fr":
            predictions = model_fr.predict(vectTrain_fr.transform(df_t['Comment']))
            comment_class = predictions[0];

        elif lan=="arabizi":
            predictions = model_arabizi.predict(vectTrain_arabizi.transform(df_t['Comment']))
            comment_class = predictions[0];

        elif lan=="arabic":
            predictions = model_arabic.predict(vectTrain_arabic.transform(df_t['Comment']))
            comment_class = predictions[0];

        else:
            df = pd.read_csv('/home/Hooriya/FinalDataSetStopWordRemoval.csv')



        return Response({'function': 'sentiment','Comment:Class':   comment_class})
    except Exception as e:
        return Response({'function': 'sentiment','result': 'there was an error ' + str(e)})


