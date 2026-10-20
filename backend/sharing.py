"""Render a private, aggregate monthly card using the same ledger as the UI."""
import calendar
from io import BytesIO
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def export_font(size):
    candidates = [os.getenv("EXPORT_FONT", ""), "C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    raise RuntimeError("한글 내보내기 글꼴이 없습니다. EXPORT_FONT 또는 fonts-noto-cjk를 설치하세요.")


def month_card(summary, image_format="JPEG", include_memo=False):
    scale = 2
    image = Image.new("RGB", (1080*scale, 1350*scale), "#f6f5ef")
    draw = ImageDraw.Draw(image)
    fonts = {}
    def text(x,y,value,size=24,fill="#243d36",max_width=None):
        if size not in fonts: fonts[size]=export_font(size*scale)
        font=fonts[size]
        while max_width and draw.textlength(str(value),font=font)>max_width*scale and size>11:
            size-=1
            if size not in fonts:fonts[size]=export_font(size*scale)
            font=fonts[size]
        draw.text((x*scale,y*scale),str(value),font=font,fill=fill)
    def box(coords,fill,radius=24):
        draw.rounded_rectangle(tuple(int(v*scale) for v in coords),radius=radius*scale,fill=fill)
    def wrapped(value,width=900,size=21):
        if size not in fonts: fonts[size]=export_font(size*scale)
        font=fonts[size]
        lines,line=[],""
        for char in value:
            if char == "\n":
                lines.append(line);line="";continue
            if draw.textlength(line+char,font=font)>width*scale:
                lines.append(line);line=char
            else: line+=char
        lines.append(line)
        return lines if len(lines)<=6 else lines[:5]+[lines[5][:-1]+"…"]
    year,month=map(int,summary["month"].split("-"))
    text(64,45,"차곡",28)
    text(64,102,f"{year}년 {month}월, 나의 소비 기록",44)
    text(66,172,"작은 기록이 모여, 나다운 한 달이 되었어요.",21, "#71827b")
    box((60,225,1020,360),"#dcebe1")
    text(87,246,"이번 달 총 지출",21)
    text(84,281,f"{summary['total']:,}원",43,max_width=610)
    text(750,253,"기록한 영수증",20)
    text(751,287,f"{summary['count']}장",35)
    top=397
    weekdays=["일","월","화","수","목","금","토"]
    width=960/7
    days={day["date"]:day for day in summary["days"]}
    first=(calendar.monthrange(year,month)[0]+1)%7
    last=calendar.monthrange(year,month)[1]
    row_count=(first+last+6)//7
    height=79 if row_count==6 else 92
    for column,label in enumerate(weekdays):
        text(78+column*width,top,label,22,"#c76e6c" if column==0 else "#6b8fbe" if column==6 else "#7e8b83")
    for index in range(row_count*7):
        day=index-first+1
        if day<1 or day>last:continue
        column,row=index%7,index//7
        x,y=60+column*width,top+43+row*height
        fill="#fcf0ec" if column==0 else "#edf3fb" if column==6 else "#ffffff"
        box((x+3,y,x+width-3,y+height-5),fill,12)
        text(x+14,y+9,day,19,"#bd6966" if column==0 else "#6187b7" if column==6 else "#62746b")
        record=days.get(f"{summary['month']}-{day:02d}")
        if record:
            text(x+12,y+38,f"{record['total']:,}",18,max_width=width-24)
    bottom=top+43+row_count*height+20
    box((60,bottom,1020,bottom+105),"#f0e6d8")
    category=summary.get("top_category")
    top_day=summary.get("top_day")
    text(85,bottom+17,"가장 많이 쓴 곳",18,"#887c68")
    text(85,bottom+48,category["category"] if category else "차곡차곡 기록 중",26)
    text(578,bottom+17,"가장 지출이 컸던 날",18,"#887c68")
    text(578,bottom+48,f"{int(top_day['date'][-2:])}일 · {top_day['total']:,}원" if top_day else "아직 기록이 없어요",24,max_width=410)
    note_top=bottom+135
    if include_memo and summary.get("memo"):
        text(66,note_top,"이번 달의 한마디",20,"#728578")
        for index,line in enumerate(wrapped(summary["memo"],930,18)):
            text(66,note_top+37+index*25,line,18)
    else:
        text(66,note_top+10,"완벽하지 않아도 괜찮아. 기록한 나에게 박수!",23)
    text(66,1285,"CHAGOK  /  MY MONTHLY MONEY DIARY",15,"#8b988e")
    notices=[]
    if summary["unknown_amount_count"]:notices.append(f"금액 미확인 {summary['unknown_amount_count']}건 제외")
    if summary.get("demo_count"):notices.append(f"샘플 {summary['demo_count']}건 포함")
    if notices:text(66,1256," · ".join(notices),15,"#9c7c62")
    output=BytesIO()
    image.save(output,format=image_format,quality=95)
    return output.getvalue()
