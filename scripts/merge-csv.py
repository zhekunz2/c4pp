import pandas as pd
import sys


def update_table(output_file, input_file):
    import pandas as pd
    from pandas.errors import EmptyDataError
    try:
        df_in = pd.read_csv(input_file, index_col='program')
        newdf_in = df_in
        #newdf_in.set_index('program', inplace=True)
    except Exception as e:
        print(e)
        return

    try:
        df = pd.read_csv(output_file, index_col='program')
        #print(df)
        #data = pd.Series(data)
        #data["program"] = data["program"].split('/')[-1]

        # if not (data["program"] == df["program"]).any():
        # namelist = list(df)
        # namelist.remove('program')
        # order = ['program']
        # order.extend(namelist)
        # df[order].to_csv(output_file)
        # newdf = pd.DataFrame(data).transpose()
        # newdf.set_index('program', inplace=True)
        df = df.drop(newdf_in.iloc[0].name, errors='ignore')

        df = df.append(newdf_in)
        df.to_csv(output_file)
        print("Outputting...")
    except Exception as e:
        print("new file...")
        #data = pd.Series(data)
        #data["program"] = data["program"].split('/')[-1]
        #data.to_csv(output_file)
        #df = pd.DataFrame(data).transpose()
        #df.set_index('program', inplace=True)
        newdf_in.to_csv(output_file)


if __name__ == '__main__':
    print(sys.argv[1])
    print(sys.argv[2])
    update_table(sys.argv[1], sys.argv[2])