#  #encoding=utf8

#  #encoding=utf8
#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pandas as pd


def group_user_interactions_csv(in_csv, out_csv, label='label', sep='\t'):
    print('group_user_interactions_csv', out_csv)
    all_data = pd.read_csv(in_csv, sep=sep)
    group_inters = group_user_interactions_df(in_df=all_data, label=label)
    group_inters.to_csv(out_csv, sep=sep, index=False)
    return group_inters


def group_user_interactions_df(in_df, label='label', seq_sep=','):
    all_data = in_df
    if label in all_data.columns:
        all_data = all_data[all_data[label] > 0]
    uids, inters = [], []
    for name, group in all_data.groupby('uid'):
        uids.append(name)
        inters.append(seq_sep.join(group['iid'].astype(str).tolist()))
    group_inters = pd.DataFrame()
    group_inters['uid'] = uids
    group_inters['iids'] = inters
    return group_inters
