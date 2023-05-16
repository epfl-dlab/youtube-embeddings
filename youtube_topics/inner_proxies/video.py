import jmespath
import logging
import traceback
import time

from .multiprocessing import MaxRunException


def get_caption(client, vid_id, tries=3):
    
    retval = None
    while retval is None and tries > 0:
        try:
            data = client.player(vid_id)
            retval = jmespath.search('captions.playerCaptionsTracklistRenderer.captionTracks[0]'
                                     '.[baseUrl, languageCode]',  data)

            if retval is None:
                break
                
        except MaxRunException:
            raise
        except Exception as e:
            logging.warning(f'Error on {vid_id} : {traceback.format_exc()}')
            time.sleep(10)
            
        finally:
            tries -= 1
        
    return retval