import pandas as pd
import sys
import glob
import os
import ast
if __name__ == '__main__':
    print(sys.argv[1])
    
    print(sys.argv[2])
    print(sys.argv[3])
    #df = pd.concat([pd.read_csv(x, index_col='program') for x in glob.glob('{0}/*/*/{1}.csv'.format(sys.argv[1], sys.argv[2])) if os.path.getsize(x) > 0])


        #df = pd.DataFrame([ast.literal_eval(open(x).read()) for x in glob.glob('{0}/*/*/{1}.csv'.format(sys.argv[1], sys.argv[2]))[i:i+1000] if os.path.getsize(x) > 0])
    df = pd.concat([pd.read_csv(x) for x in
                       glob.glob('{0}/*/*/{1}.csv'.format(sys.argv[1], sys.argv[2])) if
                       os.path.getsize(x) > 0])
    df = df.set_index('program')
    df.to_csv(sys.argv[3])
