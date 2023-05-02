

def clean_sentence(caption):
    caption = ' '.join(caption.replace(',', ' , ').split())

    # some error or typo in ScanRefer.
    caption = ' '.join(caption.replace("'m", "am").split())
    caption = ' '.join(caption.replace("'s", "is").split())
    caption = ' '.join(caption.replace("2-tiered", "2 - tiered").split())
    caption = ' '.join(caption.replace("4-drawers", "4 - drawers").split())
    caption = ' '.join(caption.replace("5-drawer", "5 - drawer").split())
    caption = ' '.join(caption.replace("8-hole", "8 - hole").split())
    caption = ' '.join(caption.replace("7-shaped", "7 - shaped").split())
    caption = ' '.join(caption.replace("2-door", "2 - door").split())
    caption = ' '.join(caption.replace("3-compartment", "3 - compartment").split())
    caption = ' '.join(caption.replace("computer/", "computer /").split())
    caption = ' '.join(caption.replace("3-tier", "3 - tier").split())
    caption = ' '.join(caption.replace("3-seater", "3 - seater").split())
    caption = ' '.join(caption.replace("4-seat", "4 - seat").split())
    caption = ' '.join(caption.replace("theses", "these").split())
    
    # some error or typo in NR3D.
    # if anno['dataset'] == 'nr3d':
    caption = ' '.join(caption.replace('.', ' .').split())
    caption = ' '.join(caption.replace(';', ' ; ').split())
    caption = ' '.join(caption.replace('-', ' ').split())
    caption = ' '.join(caption.replace('"', ' ').split())
    caption = ' '.join(caption.replace('?', ' ').split())
    caption = ' '.join(caption.replace("*", " ").split())
    caption = ' '.join(caption.replace(':', ' ').split())
    caption = ' '.join(caption.replace('$', ' ').split())
    caption = ' '.join(caption.replace("#", " ").split())
    caption = ' '.join(caption.replace("/", " / ").split())
    caption = ' '.join(caption.replace("you're", "you are").split())
    caption = ' '.join(caption.replace("isn't", "is not").split())
    caption = ' '.join(caption.replace("thats", "that is").split())
    caption = ' '.join(caption.replace("doesn't", "does not").split())
    caption = ' '.join(caption.replace("doesnt", "does not").split())
    caption = ' '.join(caption.replace("itis", "it is").split())
    caption = ' '.join(caption.replace("left-hand", "left - hand").split())
    caption = ' '.join(caption.replace("[", " [ ").split())
    caption = ' '.join(caption.replace("]", " ] ").split())
    caption = ' '.join(caption.replace("(", " ( ").split())
    caption = ' '.join(caption.replace(")", " ) ").split())
    caption = ' '.join(caption.replace("wheel-chair", "wheel - chair").split())
    caption = ' '.join(caption.replace(";s", "is").split())
    caption = ' '.join(caption.replace("tha=e", "the").split())
    caption = ' '.join(caption.replace("it’s", "it is").split())
    caption = ' '.join(caption.replace("’s", " is").split())
    caption = ' '.join(caption.replace("isnt", "is not").split())
    caption = ' '.join(caption.replace("Don't", "Do not").split())
    caption = ' '.join(caption.replace("arent", "are not").split())
    caption = ' '.join(caption.replace("cant", "can not").split())
    caption = ' '.join(caption.replace("you’re", "you are").split())
    caption = ' '.join(caption.replace('!', ' !').split())
    caption = ' '.join(caption.replace('id the', ' , the').split())
    caption = ' '.join(caption.replace('youre', 'you are').split())

    caption = ' '.join(caption.replace("'", ' ').split())

    if caption[0] == "'":
        caption = caption[1:]
    if caption[-1] == "'":
        caption = caption[:-1]
        
    return caption
