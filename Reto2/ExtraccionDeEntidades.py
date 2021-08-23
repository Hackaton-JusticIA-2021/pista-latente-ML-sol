
import pandas as pd
import numpy as np
import nltk
import os
import cv2
import imutils
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from IPython.display import clear_output, display
import time

# Montamos el Drive al Notebook
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

nltk.download("punkt")
nltk.download('cess_esp')
nltk.download('stopwords')

os.chdir("/content/drive/My Drive/Hackaton2021/codigo/Entregables/Reto2/")

import spaghetti as sgt

#@title Funciones
def split_text(BigString):
    """Split texto completo por retornos de carro o signos de puntuacion."""
    pruebat = BigString
    splited = re.split("[\,\.]\n", pruebat)
    return splited

def etiqueta_RIIA(word):
    """Etiquetar palabras completas con cadenas posibles"""
    try:
        expr = re.compile(".*{0}.*".format(word))
        busca_coincidencia = lambda lista, expr: list(filter(lambda x: expr.match(x), lista))
        newtag = []
        for optiontag, lista in zip(["per", "per", "pla", "org"] , [listProsecuted, listcivilservs, listplaces, listorgs]):
            if any(busca_coincidencia(lista, expr)) and optiontag not in newtag:
                newtag.append(optiontag)
        if len(newtag) == 0:
            newtag = ["dato"]
    except Exception as error:
        print(error)
        print("Causada por:", word)
        newtag = ["Err"]
    finally:
        return "".join(newtag)

def etiqueta_simbolo(word):
    """Etiquetar palabras que no hayan sido etiquetadas pos corpus."""
    numeric_expr = re.compile("\d+$")
    alphanum_expr = re.compile("[\w\d]+")
    char_expr = re.compile("\w+$")
    symbol_expr = re.compile("\W*.*")
    if numeric_expr.match(word) is not None:
        newtag = "numero"
    elif char_expr.match(word) is not None:
        newtag = "plbr"
    elif alphanum_expr.match(word) is not None:
        newtag = "datoN"
    elif symbol_expr.match(word) is not None:
        newtag = "unknown"
    else:
        newtag = None
    return newtag

def etiqueta_entidades_RIIA(word, currtag):
    """Seleccion de etiqueta de simbolo o palabra en RIIA."""
    if (currtag is None) and (len(word) >= 4):
        newtag = etiqueta_RIIA(word)
    else:
        newtag = etiqueta_simbolo(word)
    return newtag

def tagging(phrase):
    """Generar tags para palabras de una frase."""
    limpiar = lambda x: re.sub("[*+/\-_\\\?\'\\\n\|]", "", x)
    phrase = limpiar(phrase)
    tokens = nltk.word_tokenize(phrase)
    # limpiar palabras raras
    norare = lambda x: re.search(r"[^a-zA-ZÀ-ÿ\d]", x) is None or len(x) > 3
    # quitar stopwords
    noincluir = stopwords.words("spanish")
    seincluye = lambda x: ((x not in noincluir) or (x.isupper() or x.istitle())) and (norare(x))
    tokens = list(filter(lambda x: seincluye(x), tokens))
    tokens_low = list(map(lambda x: x.lower(), tokens))
    tagged = sgt.pos_tag(tokens_low)
    # filtrar los que resulten None
    result = []
    for (word, tag), word_unch in zip(tagged, tokens):
        if (tag is None) or (tag == ""):
            # compararlos con las entidades que se tienen de propuesta
            newtag = etiqueta_entidades_RIIA(word, tag)
            result.append((word_unch, word, newtag))
        else:
            result.append((word_unch, word, tag))
    return result

def get_chunks(grammar, tagged0):
    """Buscar expresion en frase mediante formulas gramaticales."""
    cp = nltk.RegexpParser(grammar)
    #print(tagged0)
    tagged = list(map(lambda x: (x[1], x[2]), tagged0))
    chunked = cp.parse(tagged)
    entities = []
    get_position = lambda x: np.where(list(map(lambda y: x==y[0], tagged)))[0][0]
    entitycase = lambda ind: not(tagged0[ind][0].islower())
    entitytagRIIA = lambda x: re.match(r"(per|pla|org)\w+", x) is not None
    entitycode = lambda x: x in ["Z", "numero", "Fz", "datoN"]
    entityplbr = lambda x: x in ["plbr"]
    for i, subtree in enumerate(chunked):
        if isinstance(subtree, nltk.Tree) and subtree.label() == "NP":
            inds = list(map(lambda x: get_position(x[0]), subtree.leaves()))
            withUppercase = list(map(lambda ind: entitycase(ind), inds))
            withNumbers = list(map(lambda x: entitycode(x[1]), subtree.leaves()))
            withtagRIIA = list(map(lambda x: entitytagRIIA(x[1]), subtree.leaves()))
            withplbr = list(map(lambda x: entityplbr(x[1]), subtree.leaves()))
            tokens = list(map(lambda ind: tagged0[ind][0], inds))
            tags = list(map(lambda ind: tagged0[ind][2], inds))
            percnum = float(np.sum(withNumbers)) / len(tokens)
            percplbr = float(np.sum(withplbr)) / len(tokens)
            if (percnum > 0.3) or (percplbr >= 0.5):
                entities.append(("numb", {"value":" ".join(tokens), "tags": " ".join(tags)}))
            elif any(withUppercase) or np.sum(withtagRIIA) >= 2:
                entities.append(("1st", {"value":" ".join(tokens), "tags": " ".join(tags)}))
            else:
                entities.append(("2nd", {"value":" ".join(tokens), "tags": " ".join(tags)}))
    return entities

if __name__ == "__main__":
    #@title String fields
    filename = "./output/Evaluacion_Reto2A" #@param {type:"string"}
    fileoutput = "./output/Entities_Reto2A" #@param {type:"string"}
    tabla = pd.read_csv(f"{filename}.csv", header=None)
    strip = False #@param {type:"boolean"}
    tabla

    grammar = r"""Q: {<per(\w*)|(np\w+)|nc(\w+)|pla(\w*)|org(\w*)|datoN|Z|numero|Fz|plbr>}
                  NP: {<Q> <(sp\w+)|cc>* <Q>+}
                  NP: {<Q>+}
               """

    # posibles entidades
    prosecuted = pd.read_csv("./insumos/prosecuted.csv", sep="\t")
    listProsecuted = prosecuted[prosecuted.columns[0]].tolist()
    civilservs = pd.read_csv("./insumos/civilservants.csv", sep="\t")
    listcivilservs = civilservs[civilservs.columns[0]].tolist()
    places = pd.read_csv("./insumos/places.csv", sep="\t")
    listplaces = places[places.columns[0]].tolist()
    orgs = pd.read_csv("./insumos/organizations.csv", sep="\t")
    listorgs = orgs[orgs.columns[0]].tolist()

    nrows = tabla.shape[0]
    begin = time.time()
    getvalues = lambda entsarray: "\n".join(list(map(lambda x: x[1]["value"], entsarray)))
    dfout = pd.DataFrame(columns=["C1", "C2", "Imagen", "Texto", "MainEnt", "SecondEnt", "PosiblesEnt"])
    for irow, row in enumerate(tabla.values):
        clear_output(wait=True)
        c1 = row[0]
        c2 = row[1]
        imagen = row[2]
        texto = row[3]
        if strip:
            texto = texto.strip("(").strip(")")
        splited = split_text(texto)
        entidades_texto = []
        for phrase in splited:
            if phrase != "":
                tagged = tagging(phrase)
                #print("Frase:\n", phrase)
                #print("tags:\n", tagged)
                entidades = get_chunks(grammar, tagged)
                entidades_texto.extend(entidades)

        ent1dict = list(filter(lambda x: x[0] == "1st", entidades_texto))
        ent1values = getvalues(ent1dict)
        ent2dict = list(filter(lambda x: x[0] == "2nd", entidades_texto))
        ent2values = getvalues(ent2dict)
        entcodedict = list(filter(lambda x: x[0] == "numb", entidades_texto))
        entcodevalues = getvalues(entcodedict)
        newrow = {"C1": c1, "C2": c2, "Imagen": imagen, "Texto":texto,
                  "MainEnt":ent1values, "SecondEnt": ent2values,
                  "PosiblesEnt": entcodevalues}
        dfout = dfout.append(newrow, ignore_index=True)
        # print("Entidades:\n", ent1values)
        elapsed = time.time() - begin
        print("Porcentaje de avance {0:.2f}\ttiempo transcurrido {1:0.4f} s".format((irow/nrows) * 100, elapsed))
    dfout.to_csv(f"{fileoutput}.csv", header=True)
