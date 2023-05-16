import pandas as pd
import jmespath
from .inner_proxies import MaxRunException

class RecommendationExtractor:
    """Helper class for getting video recommendations"""
    
    CONT_PATH = 'onResponseReceivedEndpoints[0].appendContinuationItemsAction.continuationItems'
    DEFAULT_PATH = 'contents.twoColumnWatchNextResults.secondaryResults.secondaryResults.results'

    
    def extract_recommended(content, continuation=False):
        """Return pandas DataFrame of videoId, channelTitle, channelId from InnerTube client next"""

        initial_str = RecommendationExtractor.CONT_PATH if continuation else RecommendationExtractor.DEFAULT_PATH

        vids_str = '[*].compactVideoRenderer.[videoId, longBylineText.runs[0].text,'\
                   'longBylineText.runs[0].navigationEndpoint.browseEndpoint.browseId]'

        return pd.DataFrame(jmespath.search(initial_str + vids_str, content), columns=['videoId','channelTitle','channelId'])
    
    def get_token(content, continuation=False):
        """Get continuation token from InnerTube client next"""
    
        initial_str = RecommendationExtractor.CONT_PATH if continuation else RecommendationExtractor.DEFAULT_PATH
        
        token_str = '[-1].continuationItemRenderer.continuationEndpoint.continuationCommand.token'
        
        return jmespath.search(initial_str + token_str, content)
    
    
    def generator(client, vid_id):
        """Create recommendations generator from InnerTube client and video id"""
        continuation = False
        token = None
        while True:
            
            vids = client.next(vid_id, continuation=token)
            
            recomms = RecommendationExtractor.extract_recommended(vids, continuation=continuation)
            
            yield recomms
            
            token = RecommendationExtractor.get_token(vids, continuation=continuation)
            continuation = True

## DEPRECATED

def extract_channel(concatdf):
    
    def extract_chan_apply(s):
        identifier = s['runs'][0]
        endpoint = identifier['navigationEndpoint']['browseEndpoint']
        return identifier['text'], endpoint['browseId']

    finaldf = (concatdf['longBylineText']
            .apply(extract_chan_apply)
            .apply(lambda x: pd.Series(x, index=['channelTitle','channelId'])))
    
    return finaldf

def get_many_vids(client, vid_id, continues=0, tries=3):
    
    # extract list of potential videos from first request
    extract_default = lambda res: (res['contents']
                                ['twoColumnWatchNextResults']
                                ['secondaryResults']
                                ['secondaryResults']
                                ['results'])
    
    # extract list of potential videos from continuation request
    extract_cont = lambda res: (res['onResponseReceivedEndpoints']
                                   [0]
                                   ['appendContinuationItemsAction']
                                   ['continuationItems'])
    
    # list of potential videos to dataframe
    to_df = lambda l: pd.DataFrame(x['compactVideoRenderer'] for x in l if 'compactVideoRenderer' in x)
    
    # extract continuation token from request
    get_token = lambda ext: (ext[-1]
                             ['continuationItemRenderer']
                             ['continuationEndpoint']
                             ['continuationCommand']
                             ['token'])
    
    concatdf = None
    
    while (concatdf is None or len(concatdf) == 0) and tries > 0:
    
        try:
            # create request
            res = client.next(vid_id)

            # extract potential videos
            ext = extract_default(res)

            # list of dfs to concat
            dfs = [to_df(ext)]

            while continues > 0:

                # get continuation token
                token = get_token(ext)

                # redo request, extract, df
                res = client.next(vid_id, continuation=token)
                ext = extract_cont(res)
                dfs.append(to_df(ext))

                # next iter
                continues -= 1

            # concat dfs
            concatdf = pd.concat(dfs).reset_index(drop=True)
        except MaxRunException:
            raise 
        except Exception as e:
            #logging.info(f'Exception {e}')
            pass
        
        # reduce num tries
        tries -= 1
    
    # return empty df
    if concatdf is None or len(concatdf) == 0:
        return pd.DataFrame([], columns=['videoId','channelTitle','channelId'])
    
    # extract channelTitle, channelId
    extracted_channel_df = extract_channel(concatdf)

    # only keep videoId, channelTitle, channelId
    final_df = pd.concat((concatdf, extracted_channel_df), axis=1)[['videoId','channelTitle','channelId']]
    

    
    return final_df
    