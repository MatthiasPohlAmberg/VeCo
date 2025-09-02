import fitz
import io
import os
from PIL import Image

# Open the PDF file
pdf_file = "C:/link/to/your/file.pdf"
output_directory = "C:/Users/your_name/Downloads/"
pdf_document = fitz.open(pdf_file)

# Iterate through each page in the PDF
for page_num in range(pdf_document.page_count):
    page = pdf_document[page_num]
    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = pdf_document.extract_image(xref)
        image_data = base_image["image"]

        # You can process or save the image as needed
        image = Image.open(io.BytesIO(image_data))
        image.save(os.path.join(output_directory, f"page{page_num + 1}_image{img_index + 1}.png"))

# Close the PDF document
pdf_document.close()