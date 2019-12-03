import pandas as pd
from lockfile import LockFile

class TableWriter:
    def __init__(self, filename):
        self.filename = filename
        self.lock = LockFile(self.filename, timeout=300)

    def read(self):
        try:
            with self.lock:
                return pd.read_csv(self.filename)
        except Exception as e:
            return None

    def write2(self, data):
        import pandas as pd
        from pandas.errors import EmptyDataError
        print("Obtaining lock...")
        with self.lock:
            try:
                df = pd.read_csv(self.filename, index_col='program')
                # print(df)
                data = pd.Series(data)
                data["program"] = data["program"].split('/')[-1]
                # if not (data["program"] == df["program"]).any():
                # namelist = list(df)
                # namelist.remove('program')
                # order = ['program']
                # order.extend(namelist)
                # df[order].to_csv(output_file)
                newdf = pd.DataFrame(data).transpose()
                newdf.set_index('program', inplace=True)
                df = df.drop(data['program'], errors='ignore')

                df = df.append(newdf)
                # df = df.append(data)
                # df.set_index('program')
                # print(df)
                df.to_csv(self.filename)
                print("Outputting...")
            except EmptyDataError as e:
                print("new file...")
                data = pd.Series(data)
                data["program"] = data["program"].split('/')[-1]
                # data.to_csv(output_file)
                df = pd.DataFrame(data).transpose()
                df.set_index('program', inplace=True)
                df.to_csv(self.filename)
            except Exception as e:
                print(e.message)
                print(e)
                # df = pd.DataFrame({})
                print(data)

    def write(self, data, all_root_dir):
        try:
            with self.lock:
                df = pd.read_csv(self.filename)
                data = pd.Series(data)
                data["program"] = data["program"].split(all_root_dir)[-1]
                if not (data["program"] == df["program"]).any():
                    namelist = list(df)
                    namelist.remove('program')
                    order = ['program']
                    order.extend(namelist)
                    df[order].to_csv(self.filename, index=False)
                    df = df.append(data, ignore_index=True)
                    df.to_csv(self.filename, index=False)
        except Exception as e:
            print(e)
            # df = pd.DataFrame({})
            print(data)
