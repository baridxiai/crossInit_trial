ALL_LANG="en es fr de zh ru pt it ar ja id tr nl pl simple fa vi sv ko he ro no hi uk cs fi hu th da ca el bg sr ms bn hr sl zh_yue az sk eo ta sh lt et ml la bs sq arz af ka mr eu tl ang gl nn ur kk be hy te lv mk zh_classical als is wuu my sco mn ceb ast cy kn br an gu bar uz lb ne si war jv ga zh_min_nan oc ku sw nds ckb ia yi fy scn gan tt am"
WIKI=/home/baridxiai/massive_data/data/xnli15
WIKI_PATH=/home/baridxiai/massive_data/data/xnli15/bz2
TOKENIZE=/home/baridxiai/massive_data/data/xnli15/XLM/tools/tokenize.sh
for lg in ${ALL_LANG}; do
    WIKI_DUMP_NAME=${lg}wiki-latest-pages-articles.xml.bz2
    if ! test -f ./word_freq/${lg}freq.txt; then
        WIKI_DUMP_LINK=https://dumps.wikimedia.org/${lg}wiki/latest/$WIKI_DUMP_NAME
        wget -c $WIKI_DUMP_LINK -P $WIKI_PATH
        wikiextractor $WIKI_PATH/bz2/$WIKI_DUMP_NAME --processes 8 -q -o - \
        | sed "/^\s*\$/d" \
        | grep -v "^<doc id=" \
        | grep -v "</doc>\$" \
        | $TOKENIZE $lg > $WIKI/txt/$lg.all
        python ./gather_wordfreq.py $WIKI_PATH/${lg}wiki-latest-pages-articles.xml.bz2 > ./word_freq/${lg}freq.txt
    fi
done