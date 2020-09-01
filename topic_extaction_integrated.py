"""
Created on 18 nov 2018
@authors: lorenzo, pasquale, antonio

Updated on sep 2020
@author Cristian
"""

from __future__ import division
import re, os, string, nltk, argparse
import numpy as np
from gensim import corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from gensim.matutils import kullback_leibler
from urllib.request import urlopen
from wordcloud import WordCloud
import matplotlib.pyplot as plt

'########## FUNCTIONS ##########'

'This function create a list of all java files in the working directory <path>'
'param: path - string'
'return: list of java files - list of string'


def list_javafiles(path):
    javafiles = [os.path.join(root, name)
                 for root, dirs, files in os.walk(path)
                 for name in files
                 if name.endswith(".java")]
    return javafiles


'This function extract source code from all java files in the working directory'
'param: javafiles_list - list of string'
'param: flag - for comments or code'
'return: list of source code of all java file - list of string'


# flag == True ->  Source code of all java files

def extract_comments_or_sourcecode_from_files_list(javafiles_list):
    'Surce code of all java files'
    multiple_files = []
    for f in javafiles_list:
        multiple_files.append(extract_comments_or_sourcecode_from_file(f))
    return multiple_files


'This function extract source code from a single java file in the working directory'
'param: f - file'
'param: flag - for comments or code'
'return: source code of a java file - string'


# flag == True ->  Source code of a single java file

def extract_comments_or_sourcecode_from_file(f):
    'Source code of a single java file'
    single_file = []

    re_onelinecomment = "//.*\n"
    re_comment = "/\*+.*?\*/"
    re_onelinecomment_object = re.compile(re_onelinecomment)
    re_comment_object = re.compile(re_comment, re.DOTALL)

    current_file = open(f, 'r', encoding='utf8')
    source_text = ''.join(current_file.readlines())
    single_file.append(re.sub(re_onelinecomment_object, "", re.sub(re_comment_object, "", source_text)))
    current_file.close()
    return ''.join(single_file)


'This function splits camel case string returning a list of token'
'param: string - string'
'return: string splitted - string'


def split_camel_case(string):
    return re.sub('(?!^)([A-Z][a-z]+)', r' \1', string).split()


'This function pre-process string <doc>'
'param: doc - string'
'param: stopwords - list of string'
'return cleaned document - string'


def clean_doc(doc, stop_words):
    'Removing punctuation'
    punc_free = ''.join(word if word not in string.punctuation else ' ' for word in doc)
    'Tokenization'
    punc_free_tokenized = word_tokenize(punc_free)
    'Splitting camel case words'
    token_comments_camel_case = []
    for word in punc_free_tokenized:
        token_comments_camel_case += split_camel_case(word)

    'Tagging'
    punc_free_tagged = nltk.pos_tag(token_comments_camel_case)

    'Lemmatization: NB. if statements are necessary to map standard tags to wordnet tags'
    'Create a lemmatizer'
    lemmatizer = WordNetLemmatizer()
    doc_normalized = []
    for word in punc_free_tagged:
        # print word[0].lower()
        if word[1].lower() == 'nn' or word[1].lower() == 'nns' or word[1].lower() == 'nnp' or word[1].lower() == 'nnps':
            try:
                doc_normalized += [str(lemmatizer.lemmatize(word[0].lower(), 'n'))]
            except UnicodeDecodeError:
                pass

    'Removing stopwords'
    stop_free = " ".join([word.lower() for word in doc_normalized if
                          word.lower() not in stop_words and re.match('[a-zA-Z][a-zA-Z]{2,}', word)])
    return stop_free


def clean_html(doc_html):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', doc_html)
    return cleantext


# restituisce la pagina successiva da analizzare
# param: page_url - url della pagina corrente
# return: nex_page_link - url della pagina web successiva
# equivale a cliccare il tasto "pagina successiva"

def get_next_page_from_url(page_url):
    page = get_page_from_url(page_url)
    start = page.find("<li class=\"arrow next\">")
    if start != -1:
        end = page.find("rel=\"next\"")
        next_page_link = "http://www.jfree.org/forum/" + page[start + 66:end - 2]
        next_page_link = next_page_link.replace(
            next_page_link[next_page_link.find("&") + 1: next_page_link.find("start")], '')
        return next_page_link
    return 0


# restituisce una pagina html in formato string
# param: url - url della pagina web
# return: pg - pagina web formato stringa

def get_page_from_url(url):
    response = urlopen(url)
    pg = response.read().decode("utf-8")
    return pg


# restituisce l'insieme degli url appartenenti ai topic all'interno di una pagina
# param: page_url - url della pagina web che contiene un insieme di topic
# return: link - insieme dei link

def get_urls_from_single_page(page_url):
    page = get_page_from_url(page_url)
    start = page.find("<ul class=\"topiclist topics\">", page.find("<div class=\"forumbg\">"))
    page = page[start: page.find("</ul>", start)]
    link_index = [start_link_index.start() for start_link_index in
                  re.finditer("<div class=\"list-inner\">", page)]
    link = ["http://www.jfree.org/forum/" + page[start_link + 54:page.find("&amp;sid", start_link)] for start_link in
            link_index]
    link = [re.sub("amp;", "", l) for l in link]
    return link


# restituisce il contenuto di un topic
# param: page_url - url della pagina web che contiene il topic da estrarre
# return: content - contenuto estratto della pagina web

def get_content_from_page(page_url):
    page = get_page_from_url(page_url)
    content_index = [start_content_index.start() for start_content_index in
                     re.finditer("<div class=\"content\">", page)]
    content = [re.sub('<[^<]+?>[\\n]', ' ', page[start + 19:page.find("</div>", start)]) for start in content_index]
    return content


def generate_WordCloud(dictionary, label="wordCloud"):
    s = ""
    for k in dictionary.keys():
        s += dictionary[k] + " "

    w = WordCloud(max_font_size=50, background_color="white").generate(s)
    plt.figure()
    plt.imshow(w, interpolation="bilinear")
    plt.axis("off")
    #plt.show()

    w.to_file('./results/'+label+'.png')



'This function compute KL-divergence score vector for two topic distribution'
'param: list_ids - list of (int, int)'
'param: ldamodel of the first model - result of LdaModel function'
'param: ldamodel of the second model - result of LdaModel function'
'param: num_of_words - int'
'return kl-divergence score vector - list of float'


def compute_kl_divergence(list_ids, lda_model1, lda_model2, num_of_words):
    kl_divergence_vector = []
    'Extracting topics distributions'
    distr1 = []
    distr2 = []
    for id1, id2 in list_ids:
        for word in lda_model1.show_topic(id1[0], topn=num_of_words):
            distr1.append(word[1])
        for word in lda_model2.show_topic(id2[0], topn=NUM_WORDS):
            distr2.append(word[1])
        'Computation of kl-divergence score for current two topics distributions'
        kl_divergence_vector.append(kullback_leibler(distr1, distr2))

    return kl_divergence_vector


def plot_num_class_topic(num_of_classes_comment, num_of_classes_source, num_topic):

    # set width of bar
    barWidth = 0.25
    plt.figure(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(num_topic)
    br2 = [x + barWidth for x in br1]
    p1 = num_of_classes_comment
    p2 = num_of_classes_source

    # Make the plot
    plt.bar(br1, p1, color='grey', width=barWidth, label='documentation')
    plt.bar(br2, p2, color='DodgerBlue', width=barWidth, label='source')

    # Adding Xticks
    plt.xlabel('Topic id', fontweight='bold')
    plt.ylabel('#Classes', fontweight='bold')
    plt.xticks([r + (barWidth/2)
                for r in range(num_topic)], np.arange(num_topic))
    plt.title(label='JFreeChart', fontsize=12, fontweight='bold')
    plt.legend()
    plt.savefig('barplot.png')
    #plt.show()


def plot_num_class_with_num_topic(num_of_classes_comment, num_of_classes_source):
    barWidth = 0.25
    plt.figure(figsize=(12, 8))
    # fig = plt.subplots(figsize=(12, 8))
    label = ['0', '1', '2-3', '4-5', '6-7', '8+']
    br3 = np.arange(len(label))
    br4 = [x + barWidth for x in br3]
    p3 = num_of_classes_comment
    p4 = num_of_classes_source
    plt.bar(br3, p4, color='grey', width=barWidth, label="Comments topic on source code")
    plt.bar(br4, p3, color='DodgerBlue', width=barWidth,
            label='Comments topic on documentation')
    plt.xlabel('Topic', fontweight='bold')
    plt.ylabel('#Classes', fontweight='bold')
    plt.xticks([r + (barWidth/2)
                for r in range(len(label))], label)
    plt.title(label='JFreeChart', fontsize=12, fontweight='bold')
    plt.legend()
    plt.savefig('barplot2.png')
    #plt.show()



'########## COMPUTATION ##########'
parser = argparse.ArgumentParser(
    description='This program implements the topic-modeling technique for source code comments quality analysis')
parser.add_argument('NUM_OF_TOPICS', default=5, help='specify number of topics', type=int)
parser.add_argument('NUM_OF_WORDS', default=10, help="specify number of words per topic", type=int)
parser.add_argument('WORKING_DIR', help='specify directory of project you want analyze')
parser.add_argument('-s, --stopwords', default='stopwords.txt', dest='STOPWORDS', metavar='STOPWORDS',
                    help='specify file that contains stopwords [default="stopwords.txt"]')
args = parser.parse_args()

'Number of topics'
NUM_TOPICS = args.NUM_OF_TOPICS
'Number of words for each topic'
NUM_WORDS = args.NUM_OF_WORDS
'Current working directory'
WORKING_DIR = args.WORKING_DIR

'Import user\'s stopwords'
STOPWORDS_file_input = args.STOPWORDS

'Set of english stopwords'
STOPWORDS = stopwords.words('english')

'If exists file stopwords.txt, STOPWORDS is extended'
if STOPWORDS_file_input:
    print('Loading stopwords from file...')
    try:
        f_stopword = open(STOPWORDS_file_input, 'r')
        st_words = ''.join(f_stopword.readlines())
        STOPWORDS.extend(st_words.split())
        f_stopword.close()
    except Exception as e:
        print('Error opening stopwords file...Loading default stopwords list')
    print('Done!')

'List of file names'
print('Retrieving source files...')
java_files = list_javafiles(WORKING_DIR)
print('Done! Founded %s files.' % len(java_files))

'Corpus (comments)'
print('Extracting comments...')

data_content = []
analyzed_pages = 0
analyzed_url = 0
pages_link =[]

forum_url = "http://www.jfree.org/forum/viewforum.php?f=3"
while forum_url and analyzed_url < len(java_files):
    analyzed_pages += 1
    print("***Analizzo la pagina*** ", forum_url, " Pagina numero: ", analyzed_pages)
    urls = get_urls_from_single_page(forum_url)
    for url in urls:
        analyzed_url += 1
        pages_link.append(url)
        for c in get_content_from_page(url):
            c = clean_html(c)
            data_content.append(c)
    forum_url = get_next_page_from_url(forum_url)

print("Url analizzati: ", analyzed_url)
print("**********************DONE**********************")

'Corpus (source code)'
print('Extracting source code...')
data_source = extract_comments_or_sourcecode_from_files_list(java_files)
print('Done!')

'Cleaning documents'
print('Processing source files...')
data_contents_cleaned = [clean_doc(doc, STOPWORDS).split() for doc in data_content]
data_source_cleaned = [clean_doc(doc, STOPWORDS).split() for doc in data_source]
print('Done!')

'Creating term dictionary'
print('Creating dictionary for forum\'s topic and source code...')
dictionary_comments = corpora.Dictionary(data_contents_cleaned)
dictionary_source = corpora.Dictionary(data_source_cleaned)
print('Done!')

generate_WordCloud(dictionary_comments, "Comment_WordCloud")
generate_WordCloud(dictionary_source, "SourceCode_WordCloud")

'Converting list of documents in a Document-Term matrix'
doc_term_matrix_comments = [dictionary_comments.doc2bow(doc) for doc in data_contents_cleaned]
doc_term_matrix_source = [dictionary_source.doc2bow(doc) for doc in data_source_cleaned]

'Running and training LDA model'
print('Running and training models...')
ldamodel_comments = LdaModel(doc_term_matrix_comments, num_topics=NUM_TOPICS, id2word=dictionary_comments, passes=100)
ldamodel_source = LdaModel(doc_term_matrix_source, num_topics=NUM_TOPICS, id2word=dictionary_source, passes=100)
print('Done!', '\n')

'Calculate coherence score'
print('Calculating coherence score of the models...')
coherence_model_lda_comments = CoherenceModel(model=ldamodel_comments, texts=data_contents_cleaned,
                                              dictionary=dictionary_comments, coherence='c_v', processes=1)
coherence_lda_comments = coherence_model_lda_comments.get_coherence()
coherence_model_lda_source = CoherenceModel(model=ldamodel_source, texts=data_source_cleaned,
                                            dictionary=dictionary_source, coherence='c_v', processes=1)
coherence_lda_source = coherence_model_lda_source.get_coherence()
print('Done!', 'Coherence score for models (<forum>,<source>):', coherence_lda_comments, ',', coherence_lda_source,
      '\n')

'Generating output files'
print('Generating output files...')
f_results_comments = open('./results/results_comments.txt', 'w', encoding="utf-8")
f_results_source = open('./results/results_source.txt', 'w', encoding="utf-8")
f_results_comments.write('NUMBER OF FILES ANALYZED: ' + repr(len(pages_link)) + '\n\n')
f_results_source.write('NUMBER OF FILES ANALYZED: ' + repr(len(java_files)) + '\n\n')
f_results_comments.write('TOPICS:\n')
f_results_source.write('TOPICS:\n')

'Saving the firsts <NUM_TOPICS> most representative topics'
for idx in range(NUM_TOPICS):
    line_comments = 'Topic #' + repr(idx) + ': ' + ldamodel_comments.print_topic(idx, topn=NUM_WORDS) + '\n'
    line_source = 'Topic #' + repr(idx) + ': ' + ldamodel_source.print_topic(idx, topn=NUM_WORDS) + '\n'
    f_results_comments.write(line_comments)
    f_results_source.write(line_source)

'Compute topic distribution for each document'
f_results_comments.write('\n')
f_results_comments.write('TOPICS DISTRIBUTION:\n')
f_results_source.write('\n')
f_results_source.write('TOPICS DISTRIBUTION:\n')
'Lists for statistics'
# strongest_topics_list_comments = []
# strongest_topics_list_source = []
topics_per_class_comments = []
topics_per_class_source = []
print ("************* ", len(doc_term_matrix_comments), "**", len(pages_link))
print("***********", len(doc_term_matrix_source),"**", len(java_files))


i = 0
for doc in doc_term_matrix_comments:
    if doc:
        temp = sorted(ldamodel_comments.get_document_topics(doc, minimum_probability=0.2), key=lambda k: k[1],
                      reverse=True)
    else:
        temp = []
    temp2 = []
    for topic in temp:
        temp2.append(topic[0])
    topics_per_class_comments.append(temp2)
    # if(temp):
    #    strongest_topics_list_comments.append(temp[0])
    if pages_link[i]:
        line = 'Topic distribution for document \"' + pages_link[i] + '\":' + repr(temp) + '\n'
        f_results_comments.write(line)
        i = i + 1

i = 0
for doc in doc_term_matrix_source:
    temp = []
    if doc:
        temp = sorted(ldamodel_source.get_document_topics(doc, minimum_probability=0.2), key=lambda k: k[1],
                      reverse=True)

    temp2 = []
    for topic in temp:
        temp2.append(topic[0])
    topics_per_class_source.append(temp2)
    # if(temp):
    #    strongest_topics_list_source.append(temp[0])
    line = 'Topic distribution for document \"' + java_files[i] + '\":' + repr(temp) + '\n'
    f_results_source.write(line)
    i = i + 1

'Computing top topics ordered by coherence score'
f_results_comments.write('\n')
f_results_comments.write('TOPICS ORDERED FOR COHERENCE SCORE:\n')
f_results_source.write('\n')
f_results_source.write('TOPICS ORDERED FOR COHERENCE SCORE:\n')
top_topics_comments = ldamodel_comments.top_topics(corpus=doc_term_matrix_comments, texts=data_contents_cleaned,
                                                   dictionary=dictionary_comments, coherence='c_v', topn=NUM_WORDS,
                                                   processes=1)
top_topics_source = ldamodel_source.top_topics(corpus=doc_term_matrix_source, texts=data_source_cleaned,
                                               dictionary=dictionary_source, coherence='c_v', topn=NUM_WORDS,
                                               processes=1)
for top_topic in top_topics_comments:
    f_results_comments.write(repr(top_topic) + '\n')
for top_topic in top_topics_source:
    f_results_source.write(repr(top_topic) + '\n')

'Computing strongest represented topics (from forum) in source code'
f_results_comments.write('\n')
f_results_source.write('\n')
f_results_comments.write('DISTRIBUTION OF FORUM TOPICS ON SOURCE CODE FILES:\n')
f_results_source.write('DISTRIBUTION OF SOURCE CODE TOPICS ON FORUM FILES:\n')
doc_term_matrix_sc = [dictionary_comments.doc2bow(doc) for doc in data_source_cleaned]
doc_term_matrix_cs = [dictionary_source.doc2bow(doc) for doc in data_contents_cleaned]
strongest_topics_comments = []
strongest_topics_source = []
topics_comments_per_class = []
topics_source_per_class = []
for source_doc in doc_term_matrix_sc:
    temp = sorted(ldamodel_comments[source_doc], key=lambda k: k[1], reverse=True)
    strongest_topics_comments.append(temp)
    temp2 = []
    for topic in temp:
        if topic[1] >= 0.2:
            temp2.append(topic[0])
    topics_comments_per_class.append(temp2)
for comments_doc in doc_term_matrix_cs:
    temp = sorted(ldamodel_source[comments_doc], key=lambda k: k[1], reverse=True)
    strongest_topics_source.append(temp)
    temp2 = []
    for topic in temp:
        if topic[1] >= 0.2:
            temp2.append(topic[0])
    topics_source_per_class.append(temp2)
i = 0
for elem in strongest_topics_comments:
    line = 'Topic distribution for document \"' + java_files[i] + '\":' + repr(elem)
    f_results_comments.write(line + '\n')
    i = i + 1
i = 0
for elem in strongest_topics_source:
    line = 'Topic distribution for document \"' + java_files[i] + '\":' + repr(elem)
    f_results_source.write(line + '\n')
    i = i + 1
f_results_comments.close()
f_results_source.close()

'########## STATISTICS ##########'

'Computing KL-divergence score vector for comparison between topic'
ids_list = []
strongest_topics_list_comments = list(strongest_topics_comments)
strongest_topics_list_source = list(strongest_topics_source)
if len(strongest_topics_list_comments) == len(strongest_topics_list_source):
    for j in range(len(strongest_topics_list_comments)):
        ids_list.append((strongest_topics_list_comments[j][0], strongest_topics_list_source[j][0]))
        j = j + 1
        kl_divergence_vector = sorted(compute_kl_divergence(ids_list, ldamodel_comments, ldamodel_source, NUM_WORDS))
        mean_divergence_score = 0
        print(j)
        for elem in kl_divergence_vector:
            mean_divergence_score += elem
            mean_divergence_score = mean_divergence_score / len(kl_divergence_vector)

'Computing number of classes with: 0 topics, 1 topic, 2-3 topics, 4-5 topics, 6-7 topics, 8+ topics; (5 categories)'
'In this case we consider distribution of comments topics on comments'


def placement(topic):
    new_num_of_class = [0] * 6
    pos = 0
    for elem in topic:
        if len(elem) == 0:
            pos = 0
        elif len(elem) == 1:
            pos = 1
        elif len(elem) in range(2, 4):
            pos = 2
        elif len(elem) in range(4, 6):
            pos = 3
        elif len(elem) in range(6, 8):
            pos = 4
        else:
            pos = 5
        temp = new_num_of_class.pop(pos)
        new_num_of_class.insert(pos, temp + 1)
    return new_num_of_class


num_of_class_per_num_of_topic_comments_on_comments = placement(topics_per_class_comments)

'Computing number of classes with: 0 topics, 1 topic, 2-3 topics, 4-5 topics, 6-7 topics, 8+ topics; (5 categories)'
'In this case we consider distribution of source code topics on source code'
num_of_class_per_num_of_topic_source_on_source = placement(topics_per_class_source)

'Computing number of classes with: 0 topics, 1 topic, 2-3 topics, 4-5 topics, 6-7 topics, 8+ topics; (5 categories)'
'In this case we consider distribution of comments topics on source code'
num_of_class_per_num_of_topic_comments_on_source = placement(topics_comments_per_class)

'Computing number of classes with: 0 topics, 1 topic, 2-3 topics, 4-5 topics, 6-7 topics, 8+ topics; (5 categories)'
'In this case we consider distribution of source code topics on comments'
num_of_class_per_num_of_topic_source_on_comments = placement(topics_source_per_class)


def number_of_classes(topic):
    new_number_of_classes = [0] * NUM_TOPICS
    for elem in topic:
        for topic_id in elem:
            temp = new_number_of_classes.pop(topic_id)
            new_number_of_classes.insert(topic_id, temp + 1)
    return new_number_of_classes


'Computing number of classes for each topic (comments topics on comments)'
num_of_class_per_topic_comments_on_comments = number_of_classes(topics_per_class_comments)

'Computing number of classes for each topic (source code topics on source code)'
num_of_class_per_topic_source_on_source = number_of_classes(topics_per_class_source)

'Computing number of classes for each topic (comments topics on source code)'
num_of_class_per_topic_comments_on_source = number_of_classes(topics_comments_per_class)

'Computing number of classes for each topic (source code topics on comments)'
num_of_class_per_topic_source_on_comments = number_of_classes(topics_source_per_class)

'Generating formatted file containing results...'
file_out_stats = open('./results/stats.txt', 'w', encoding="utf-8")
file_out_stats.write('Total number of files: ' + repr(len(java_files)) + '\n\n')

if ids_list:
    file_out_stats.write('KL-DIVERGENCE SCORE BETWEEN TOPICS GENERATED BY MODELS:\n')
    file_out_stats.write('Min: ' + repr(kl_divergence_vector[0]) + '\n')
    file_out_stats.write('Max: ' + repr(kl_divergence_vector[len(kl_divergence_vector) - 1]) + '\n')
    file_out_stats.write('Mean: ' + repr(mean_divergence_score) + '\n\n')
else:
    file_out_stats.write('KL-DIVERGENCE SCORE FAILED!\n\n')


def write_topics_number(str, num_topic):
    file_out_stats.write(str)
    for elem in range(len(num_topic)):
        num_topic_str = 'Number of classes with '
        if elem == 0:
            num_topic_str += '0'
        elif elem == 1:
            num_topic_str += '1'
        elif elem == 2:
            num_topic_str += '2-3'
        elif elem == 3:
            num_topic_str += '4-5'
        elif elem == 4:
            num_topic_str += '6-7'
        else:
            num_topic_str += 'more than 8'

        num_topic_str += ' topics: ' + repr(num_topic[elem]) + '\n'
        file_out_stats.write(num_topic_str)


write_topics_number(
    'NUMBER OF CLASSES WITH A CERTAIN NUMBER OF TOPICS (CONSIDERING DISTRIBUTION OF FORUM TOPICS ON FORUM FILES):\n',
    num_of_class_per_num_of_topic_comments_on_comments)

write_topics_number(
    '\nNUMBER OF CLASSES WITH A CERTAIN NUMBER OF TOPICS (CONSIDERING DISTRIBUTION OF SOURCE CODE TOPICS ON SOURCE CODE FILES):\n',
    num_of_class_per_num_of_topic_source_on_source)

write_topics_number(
    '\nNUMBER OF CLASSES WITH A CERTAIN NUMBER OF TOPICS (CONSIDERING DISTRIBUTION OF FORUM TOPICS ON SOURCE CODE FILES):\n',
    num_of_class_per_num_of_topic_comments_on_source)

write_topics_number(
    '\nNUMBER OF CLASSES WITH A CERTAIN NUMBER OF TOPICS (CONSIDERING DISTRIBUTION OF SOURCE CODE TOPICS ON FORUM FILES):\n',
    num_of_class_per_num_of_topic_source_on_comments)


def write_stats(str, num_of_class_on):
    file_out_stats.write(str)
    for elem in range(len(num_of_class_on)):
        file_out_stats.write('Number of classes in which topic ' + repr(elem) + ' appears: ' + repr(
            num_of_class_on[elem]) + '\n')


write_stats('\nNUMBER OF CLASSES FOR EACH TOPIC (CONSIDERING FORUM TOPICS ON FORUM FILES):\n',
            num_of_class_per_topic_comments_on_comments)

write_stats('\nNUMBER OF CLASSES FOR EACH TOPIC (CONSIDERING SOURCE CODE TOPICS ON SOURCE CODE FILES):\n',
            num_of_class_per_topic_source_on_source)

write_stats('\nNUMBER OF CLASSES FOR EACH TOPIC (CONSIDERING FORUM TOPICS ON SOURCE CODE FILES):\n',
            num_of_class_per_topic_comments_on_source)

write_stats('\nNUMBER OF CLASSES FOR EACH TOPIC (CONSIDERING SOURCE CODE TOPICS ON FORUM FILES):\n',
            num_of_class_per_topic_source_on_comments)

file_out_stats.close()
print('Done! Results stored in generated files results_comments.txt, results_source.txt, stats.txt.', '\n')

plot_num_class_topic(num_of_class_per_topic_comments_on_comments, num_of_class_per_topic_source_on_source, NUM_TOPICS)
plot_num_class_with_num_topic(num_of_class_per_num_of_topic_comments_on_comments, num_of_class_per_num_of_topic_source_on_source)
