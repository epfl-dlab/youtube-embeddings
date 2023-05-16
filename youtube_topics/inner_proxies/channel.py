import pandas as pd
import jmespath
import numpy as np
from .multiprocessing import MaxRunException


class ChannelInfo:
    
    DEFAULT_STR = 'contents.twoColumnWatchNextResults.playlist.playlist'
    CONT_STR = 'continuationContents.playlistPanelContinuation'
    TOKEN_STR = '.continuations[0].nextContinuationData.continuation'
    FETCH_STR = '.contents[*].playlistPanelVideoRenderer.[title.simpleText, lengthText.simpleText, navigationEndpoint.watchEndpoint.videoId]'

    def extract_videos(content, continuation=False):
        
        initial_str = ChannelInfo.CONT_STR if continuation else ChannelInfo.DEFAULT_STR

        return pd.DataFrame(jmespath.search(initial_str + ChannelInfo.FETCH_STR, content),
                            columns=['video','time','id'])
    
    def extract_token(content, continuation=False):
        
        initial_str = ChannelInfo.CONT_STR if continuation else ChannelInfo.DEFAULT_STR
        
        return jmespath.search(initial_str + ChannelInfo.TOKEN_STR, content)
        
    def video_generator(client, channel_id):
        """Create video generator from InnerTube client and channel id"""
        continuation = False
        token = None
        playlist_id = 'UU' + channel_id[2:]
        
        while True:
            
            content = client.next(playlist_id=playlist_id, continuation=token)
            
            videos = ChannelInfo.extract_videos(content, continuation=continuation)
            
            if len(videos) == 0:
                break
                
            yield videos
            
            token = ChannelInfo.extract_token(content, continuation=continuation)
            continuation = True
        
    
def time_to_sec(time_string):
    
    return sum(int(t)*(60**i) for i, t in enumerate(reversed(time_string.split(':'))))

def get_playlist_vids(client, playlist_id, continues=0, tries=3):
    """[DEPRECATED] Fetch videos from playlist
    
    Use ChannelInfo instead

    Args:
        client (InnerTube): InnerTube client for fetching
        playlist_id (str): Playlist id for which to fetch videos
        continues (int, optional): Number of batches to fetch (by default, only gets the first one). Defaults to 0.
        tries (int, optional): Number of retries in case of error. Defaults to 3.

    Returns:
        pd.DataFrame: Videos from playlist
    """
    
    to_df = lambda play: pd.DataFrame(x['playlistPanelVideoRenderer'] for x in play['contents'] if 'playlistPanelVideoRenderer' in x)
    get_token = lambda play_items: play_items['continuations'][0]['nextContinuationData']['continuation']
    
    concatdf = None
    
    #logging.info('BeforeCall')
    while (concatdf is None or len(concatdf) == 0) and tries > 0:
        
        try:
    
            # fetch playlist
            playlist = client.next(playlist_id=playlist_id)
            

            # get items, token
            playlist_items = playlist['contents']['twoColumnWatchNextResults']['playlist']['playlist']

            # list of dfs to concat
            dfs = [to_df(playlist_items)]

            while continues > 0:

                # get continuation token
                token = get_token(playlist_items)

                # redo request, extract, df
                res = client.next(playlist_id, continuation=token)
                playlist_items = res['continuationContents']['playlistPanelContinuation']
                dfs.append(to_df(playlist_items))

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
        
    #logging.info('Aftercall')
    
    
    # return empty df
    if concatdf is None or len(concatdf) == 0:
        return pd.DataFrame([], columns=['videoId','channelTitle','channelId', 'videoTitle', 'playTime'])
    
    try:
        concatdf['videoTitle'] = concatdf['title'].apply(lambda x: x['simpleText'])
    except MaxRunException:
            raise
    except Exception as e:
        #logging.info(f'Exception {e}')
        concatdf['videoTitle'] = np.NaN
        
    try:
        concatdf['playTime'] = concatdf['lengthText'].apply(lambda x: 0 if not isinstance(x, dict) else time_to_sec(x['simpleText']))
    except MaxRunException:
            raise
    except Exception as e:
        #logging.info(f'Exception {e}')
        concatdf['playTime'] = np.NaN
    
    return concatdf
    