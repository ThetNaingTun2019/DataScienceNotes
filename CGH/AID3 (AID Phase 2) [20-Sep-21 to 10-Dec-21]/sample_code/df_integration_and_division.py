# 1_複数ファイルの読み込みと統合
# Count File Numbers
data_dir = '../../Step2/output_pkl/'
data_filelist = os.listdir(data_dir)
file_num = len(data_filelist)
print("file_num:",file_num)

# Read Data and Integrate into 1 Data Frame
df_raw = pd.DataFrame()
for i_file in range(file_num):
    file_dir = data_dir + "/" + data_filelist[i_file]
    df_tmp = pd.read_pickle(file_dir)
    df_raw = pd.concat([df_raw, df_tmp])
df_raw.reset_index(drop=True, inplace=True)
print("df_raw.shape:", df_raw.shape)


# 2_分割して保存
output_file_num = 10 #Separation of df_training
df_total_row = df_raw.shape[0]//output_file_num
for j in range (output_file_num):
    print(df_raw[j*df_total_row:(j+1)*df_total_row])
    df_raw[j*df_total_row:(j+1)*df_total_row].to_pickle('df_raw_{}.pkl'.format(j))