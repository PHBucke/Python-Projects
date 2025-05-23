import cv2
import imutils
import numpy as np
import argparse

def detect(frame):
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    
    person = 1
    
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1
    
    cv2.putText(frame, 'Status: Detectando ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total de pessoas: {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('output', frame)

    return frame

def detect_by_path_video(path, writer):

    video = cv2.VideoCapture(path)
    check, frame = video.read()
    
    if check == False:
        print('Não foi possível encontrar vídeo. Insira um caminho válido.')
        return

    print('Detectando pessoas...')
    
    while video.isOpened():
        check, frame =  video.read()

        if check:
            frame = imutils.resize(frame , width = min(800, frame.shape[1]))
            frame = detect(frame)
            
            if writer is not None:
                writer.write(frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

def detect_by_camera(writer):   
    video = cv2.VideoCapture(0)
    print('Detectando pessoas...')

    while True:
        check, frame = video.read()

        frame = detect(frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()

def detect_by_path_image(path, output_path):
    image = cv2.imread(path)

    image = imutils.resize(image, width = min(800, image.shape[1])) 

    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def human_detector(args):
    image_path = args["image"]
    video_path = args['video']
    if str(args["camera"]) == 'true' :
        camera = True 
    else :
        camera = False

    writer = None
    output_path = None
    
    if args['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))

    if camera:
        print('[INFO] Abrindo a câmera.')
        detect_by_camera(output_path, writer)
        
    elif video_path is not None:
        print('[INFO] Abrindo o vídeo do caminho selecionado.')
        detect_by_path_video(video_path, writer)
        
    elif image_path is not None:
        print('[INFO] Abrindo imagem do caminho selecionado.')
        detect_by_path_image(image_path, args['output'])

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default = None, help = "Caminho para o arquivo de vídeo.")
    arg_parse.add_argument("-i", "--image", default = None, help = "Caminho para a imagem.")
    arg_parse.add_argument("-c", "--camera", default = False, help = "Mude para True se você quer a câmera seja acionada.")
    arg_parse.add_argument("-o", "--output", type = str, help = "Caminho para um arquivo de vídeo diferente.")
    args = vars(arg_parse.parse_args())

    return args

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = argsParser()
    human_detector(args)