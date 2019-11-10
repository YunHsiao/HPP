if [ $# -ne 1 ]; then
    echo "usage: $0 <mp number>"
    exit 1
fi

MP=$1
TARGET=build/Debug/MP${MP}.exe

if [ ! -x ${TARGET} ]; then
    echo "unable to find executable ${TARGET}"
    exit 1
fi

DATADIR=data/${MP}/data

if [ ! -x ${DATADIR} ]; then
    echo "unable to find directory ${DATADIR}"
    exit 1
fi

case ${MP} in
0|1|4|5|7|8|9|12)
    TYPE=vector
    ;;
2|3)
    TYPE=matrix
    ;;
6|11)
    TYPE=image
    ;;
*)
    echo "unknown type for MP${MP}"
    exit 1
esac

for dataset in ${DATADIR}/*; do
    echo "executing with dataset ${dataset}"
    inputs=`ls ${dataset}/input* 2>/dev/null | xargs echo | sed -e "s/  */,/g"`
    expected=`ls ${dataset}/output* 2>/dev/null `

    if [ -z ${inputs} ]; then inputs=none; fi

    if [ -z ${expected} ]; then expected=none; fi

    if [ ${TYPE} == "image" ]; then ext=ppm;
    else ext=raw; fi

    echo ">> ${TARGET} -e ${expected} -i ${inputs} -o ${dataset}/result.${ext} -t ${TYPE}"
    ${TARGET} -e ${expected} -i ${inputs} -o ${dataset}/result.${ext} -t ${TYPE} |
        grep -oP --color 'message\"\s*\:\s*\"\s*\K(.*(correct)?.*)(?=\")'
done
