class MetaSentence:
    data = []
    
    def __init__(self, sentence):
	    self.data = sentence
		
    def strip_metadata(self):
	    if len(self.data) > 0:
	        return [x[0] for x in self.data]

    def get_tag(self, word):
        for meta_tuple in self.data:
            if meta_tuple[0] == word:
                return meta_tuple[1]
        return None
        
    def get_data(self, word):
        for meta_tuple in self.data:
            if meta_tuple[0] == word:
                return meta_tuple
        return None