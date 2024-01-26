# parses .meta4 file to download tiff files
import os
import requests
from xml.etree import ElementTree as ET
from multiprocessing.pool import ThreadPool



# dir to metadata
resources_dir = 'resources'
# dir to store raster data
raster_data_dir = 'raster_data'

if not  os.path.exists(raster_data_dir):
   # Create a new directory because it does not exist
   os.makedirs(raster_data_dir)
   print("The new directory is created!")


urls_to_download = []
total_size = 0

def download_file(data):
    global number_of_files
    destination, url = data
    print(url)
    destination = os.path.join(raster_data_dir, destination)
    print(f"Downloading {destination}")
    with requests.get(url, stream=True, timeout=60) as response:
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    number_of_files -= 1
    print(number_of_files)
    # print(f"Finished {destination}")

def download_metalink(metadata):
        global total_size
        global number_of_files
        root = ET.fromstring(metadata)

        for file_element in root.findall(".//{urn:ietf:params:xml:ns:metalink}file"):
            file_name = file_element.get("name")
            file_size = int(file_element.find("{urn:ietf:params:xml:ns:metalink}size").text)
            total_size += file_size

            for url_element in file_element.findall("{urn:ietf:params:xml:ns:metalink}url"):
                file_url = url_element.text
                print(f'{file_name}:{file_url}')
                try:
                    filename = os.path.join(raster_data_dir, file_name)
                    if os.path.exists(filename):
                        if os.path.getsize(filename) != file_size:
                        # download_file(file_url, os.path.join(output_directory, file_name))
                        # print(f"Download successful for {file_name}")
                            urls_to_download.append([file_name, file_url])

                    else:
                        urls_to_download.append([file_name, file_url])

                    break  # If download is successful from one URL, break out of the loop
                except Exception as e:
                    print(f"Error downloading {file_name} from {file_url}: {e}")


def read_file(filename):
    # Read the entire content of the file into a string
    # Open the file in read mode
    with open(filename, 'r') as file:
        # Read the entire content of the file into a string
        file_content = file.read()


    download_metalink(file_content)


metadata_files = os.listdir(resources_dir)

for filename in metadata_files:
    if '97' in filename:
        read_file(os.path.join(resources_dir, filename))
print(total_size)
gigabytes_value = total_size / (1024 ** 2)

print(f"{total_size} kilobytes is approximately {gigabytes_value:.2f} gigabytes.")

number_of_files = len(urls_to_download)
# download in parallel
p = ThreadPool(10)
xs = p.map(download_file, urls_to_download)

# if __name__ == "__main__":
#     metalink_url = "YOUR_METALINK_URL_HERE"
#     output_directory = "YOUR_OUTPUT_DIRECTORY_HERE"
#
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)

# download_metalink(metalink_url, output_directory)