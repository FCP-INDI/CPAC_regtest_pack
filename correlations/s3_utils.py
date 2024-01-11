
def pull_NIFTI_file_list_from_s3(s3_directory, s3_creds):

    import os
    try:
        from indi_aws import fetch_creds
    except:
        err = "\n\n[!] You need the INDI AWS package installed in order to " \
              "pull from an S3 bucket. Try 'pip install indi_aws'\n\n"
        raise Exception(err)

    s3_list = []

    s3_path = s3_directory.replace("s3://","")
    bucket_name = s3_path.split("/")[0]
    bucket_prefix = s3_path.split(bucket_name + "/")[1]

    bucket = fetch_creds.return_bucket(s3_creds, bucket_name)

    # Build S3-subjects to download
    # maintain the "s3://<bucket_name>" prefix!!
    print("Gathering file paths from {0}\n".format(s3_directory))
    for bk in bucket.objects.filter(Prefix=bucket_prefix):
        if ".nii" in str(bk.key):
            s3_list.append(os.path.join("s3://", bucket_name, str(bk.key)))

    if len(s3_list) == 0:
        err = "\n\n[!] No filepaths were found given the S3 path provided!" \
              "\n\n"
        raise Exception(err)

    return s3_list



def download_from_s3(s3_path, local_path, s3_creds):

    import os

    try:
        from indi_aws import fetch_creds, aws_utils
    except:
        err = "\n\n[!] You need the INDI AWS package installed in order to " \
              "pull from an S3 bucket. Try 'pip install indi_aws'\n\n"
        raise Exception(err)

    s3_path = s3_path.replace("s3://","")
    bucket_name = s3_path.split("/")[0]
    bucket_prefix = s3_path.split(bucket_name + "/")[1]

    filename = s3_path.split("/")[-1]
    local_file = os.path.join(local_path, filename)

    if not os.path.exists(local_file):
        bucket = fetch_creds.return_bucket(s3_creds, bucket_name)
        aws_utils.s3_download(bucket, ([bucket_prefix], [local_file]))

    return local_file