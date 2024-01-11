def create_boxplot(corr_group, corr_group_name, pipeline_names=None,
                   current_dir=None):

    import os
    import numpy as np
    from matplotlib import pyplot

    if not pipeline_names:
        pipeline_names = ["pipeline_1", "pipeline_2"]
    if not current_dir:
        current_dir = os.getcwd()

    allData = []
    labels = list(corr_group.keys())
    labels.sort()

    for label in labels:
        if "file reading problem" in label:
            print(f"File Reading Error: {label} ")
            continue
        try:
            allData.append(np.asarray(corr_group[label]).astype(float))
        except ValueError as ve:
            print(f"Value Error Ocurred. \n{ve}")
            continue
            #raise Exception(ve)

    pyplot.boxplot(allData)
    pyplot.xticks(range(1,(len(corr_group)+1)),labels,rotation=85)
    pyplot.margins(0.5,1.0)
    #pyplot.ylim(0.5,1.2)
    pyplot.xlabel('Derivatives')
    pyplot.title('Correlations between {0} and {1}\n '
                 '( {2} )'.format(list(pipeline_names)[0], 
                                  list(pipeline_names)[1],
                                  corr_group_name))

    output_filename = os.path.join(current_dir,
                                   (corr_group_name + "_" +
                                    list(pipeline_names)[0] +
                                    "_and_" + list(pipeline_names)[1]))

    pyplot.savefig('{0}.png'.format(output_filename), format='png', dpi=200, bbox_inches='tight')
    pyplot.close()