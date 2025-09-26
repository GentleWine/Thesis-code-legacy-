from PyPDF2 import PdfReader, PdfWriter

# PDF文件的路径
pdf_path1 = 'pdf/1.pdf'
pdf_path2 = 'pdf/2.pdf'

# 创建一个PDF阅读器对象
reader1 = PdfReader(pdf_path1)
reader2 = PdfReader(pdf_path2)

# 创建一个PDF写入器对象
writer = PdfWriter()

# 将第一个PDF的所有页面添加到写入器
for page in reader1.pages:
    writer.add_page(page)

# 将第二个PDF的所有页面添加到写入器
for page in reader2.pages:
    writer.add_page(page)

# 输出文件的路径
output_path = 'pdf/ZhongZhao-SparsePCA-supp.pdf'

# 将合并后的PDF写入到文件
with open(output_path, 'wb') as out:
    writer.write(out)

print("PDF文件已成功合并")