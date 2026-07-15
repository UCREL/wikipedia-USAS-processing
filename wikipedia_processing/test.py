from models_install import Languages
from models_util import get_language_sentence_splitter

print(list(get_language_sentence_splitter(Languages.en)("Being close to the Arid Diagonal of South America, the mountain has extremely dry conditions, which prevent the formation of substantial glaciers and a permanent snow cover. Despite the arid climate, there is a permanent crater lake about 100 m (330 ft) in diameter at an elevation of 6,480–6,500 metres (21,260–21,330 ft) within the summit crater and east of the main summit. This is the highest lake of any kind in the world. Owing to its altitude and the desiccated climate, the mountain lacks vegetation.")))
#for language in Languages:
    #value = get_language_sentence_splitter(language)
    #value.analyze_pipes(pretty=True)
    #print(f"{value.lang} -- {value.pipe_names}")
    