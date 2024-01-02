# -*- coding: utf-8 -*-
"""
******************************************************************************
Created on Sun Jan  2 16:06:01 2022

@author: valerie Desnoux

d'apres https://github.com/grmarcil/image_spline/blob/master/spline.py
et papier Burt and Adelson's "A Multiresolution Spline With Application to Image Mosaics"

******************************************************************************
Version 1.1
- ne pas oublier de mettre dans le bon ordre les tableaux y1,y2,x1, x2
Version 1.0
- traduction anglaise
- gestion erreur format fichiers, dimensions

"""

import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageMath
import PySimpleGUI as sg
import os
import io
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2
import Inti_functions as inti
import blend_functions as bl
import scipy
import scipy.ndimage as ndimage
import cv2

import sys
import yaml

# -------------------------------------------------------------
global LG # Langue de l'interfacer (1 = FR ou 2 = US)
LG = 2
# -------------------------------------------------------------


current_version = 'Inti_Mosaic V1.1 by V.Desnoux '

def im_reduce(img):
    '''
    Apply gaussian filter and drop every other pixel
    '''
    filter = 1.0 / 20 * np.array([1, 5, 8, 5, 1])
    lowpass = ndimage.filters.correlate1d(img, filter, 0)
    lowpass = ndimage.filters.correlate1d(lowpass, filter, 1)
    im_reduced = lowpass[::2, ::2, ...]
    return im_reduced

def gaussian_pyramid(image, layers):
    '''
    pyramid of increasingly strongly low-pass filtered images,
    shrunk 2x h and w each layer
    '''
    pyr = [image]
    temp_img = image
    for i in range(layers):
        temp_img = im_reduce(temp_img)
        pyr.append(temp_img)
    return pyr

def fits_to_PIL (myimg, s1,s2):
    myimg_s=seuil_image(myimg, s1, s2)
    myimg8=cv2.convertScaleAbs(myimg_s,alpha=255/(65535), beta=0.1)
    im = Image.fromarray(myimg8)
    return im
    
def seuil_image (i, Seuil_bas, Seuil_haut):
    img=np.copy(i)
    img[img>Seuil_haut]=Seuil_haut
    if Seuil_haut!=Seuil_bas :
        img_seuil=(img-Seuil_bas)* (65535/(Seuil_haut-Seuil_bas))
        img_seuil[img_seuil<0]=0
    else:
        img_seuil=img
    return img_seuil

def img_resize_frompng (nomfich,dim):
    image = Image.open(nomfich)
    image.thumbnail((dim, dim))
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    out=bio.getvalue()
    #with io.BytesIO() as output:
        #image.save(output, format="PNG")
        #out = output.getvalue()
    return out

def img_resize (im,li,co):
    im.thumbnail((li, co))
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    out=bio.getvalue()
    return out

def img_paste (result, im1,im2, dx, dy):
    result.paste(im1, (0,0)) #left, top
    result.paste(im2, (dx,dy))
    
def trad (textFR, textEN):
    if LG == 1 :
        print(textFR)
    else:
        print(textEN)
    return


"""
-----------------------------------------------------------------------------------------------
Programme de mosaic de deux images 
-----------------------------------------------------------------------------------------------
"""

debug=False

#recupere les parametres utilisateurs enregistrés lors de la session
#precedente dans un fichier ini
my_dictini={'obs_dir':''}
try:
    my_ini=os.path.dirname(sys.argv[0])+'/mosa_ini.yaml'
    #print('mon repertoire : ', my_ini)
    with open(my_ini, "r") as f1:
        my_dictini = yaml.safe_load(f1)
    WorkDir=my_dictini['obs_dir']
except:
    WorkDir=''
    


#fenetre pour recuperer le nom du fichier et le repertoire
sg.theme('Dark2')
sg.theme_button_color(('white', '#500000'))
tdim=65
gdim=600
win_pos=(200,10)

colonne0 = [
        [sg.Graph(canvas_size=(gdim,gdim),graph_bottom_left=(0, 0),graph_top_right=(gdim,gdim),drag_submits=True, enable_events=True,key='-MOSA-', background_color="darkgrey")]
        ]
    
colonne1 = [
        [sg.Button ('Run', size=(10,1),disabled=True)],
        [sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],
        [sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],[sg.Text('')],
        [sg.Text('')],[sg.Text('')],[sg.Text('')],
        [sg.Button("Quit", size=(10,1))]
        ]       

if LG ==1 : # en francais
    
    layout = [
        [sg.Text('Fichiers :', size=(8, 1)), sg.InputText(default_text='',size=(75,1),enable_events=True,key='-FILE1-'),
         sg.FilesBrowse('Ouvrir',size=(10,1), file_types=(("Fits Files", "*.fits"),),initial_folder=WorkDir)],
        #[sg.Text("Valeurs rayons", key="-RAYONS-")],
        #[sg.Text('Rayon de référence :', size=(16,1)), sg.InputText(default_text='1520', size=(8,1), key="-RAYREF-")],
        [sg.Graph(canvas_size=(tdim,tdim),graph_bottom_left=(0, 0),graph_top_right=(tdim,tdim),drag_submits=True, enable_events=True,key='-THUMB1-'),
         sg.Graph(canvas_size=(tdim,tdim),graph_bottom_left=(0, 0),graph_top_right=(tdim,tdim),drag_submits=True, enable_events=True,key='-THUMB2-'),
         sg.Graph(canvas_size=(tdim,tdim),graph_bottom_left=(0, 0),graph_top_right=(tdim,tdim),drag_submits=True, enable_events=True,key='-THUMB3-')],
         #sg.Graph(canvas_size=(tdim,tdim),graph_bottom_left=(0, 0),graph_top_right=(tdim,tdim),drag_submits=True, enable_events=True,key='-THUMB4-'),
         #sg.Graph(canvas_size=(tdim,tdim),graph_bottom_left=(0, 0),graph_top_right=(tdim,tdim),drag_submits=True, enable_events=True,key='-THUMB5-'),
         #sg.Button("Recalage fin",size=(10,1), disabled=True),
        [sg.Column(colonne0),sg.Column(colonne1)],
        [sg.Text(current_version, size=(30, 1),text_color='Tan', font=("Arial", 8, "italic"))]
        ]
else:   # en anglais
    layout = [
        [sg.Text('Files :', size=(8, 1)), sg.InputText(default_text='',size=(75,1),enable_events=True,key='-FILE1-'),
         sg.FilesBrowse('Open',size=(10,1),file_types=(("Fits Files", "*.fits"),),initial_folder=WorkDir)],
        #[sg.Text("Valeurs rayons", key="-RAYONS-")],
        #[sg.Text('Rayon de référence :', size=(16,1)), sg.InputText(default_text='1520', size=(8,1), key="-RAYREF-")],
        [sg.Graph(canvas_size=(tdim,tdim),graph_bottom_left=(0, 0),graph_top_right=(tdim,tdim),drag_submits=True, enable_events=True,key='-THUMB1-'),
         sg.Graph(canvas_size=(tdim,tdim),graph_bottom_left=(0, 0),graph_top_right=(tdim,tdim),drag_submits=True, enable_events=True,key='-THUMB2-'),
         sg.Graph(canvas_size=(tdim,tdim),graph_bottom_left=(0, 0),graph_top_right=(tdim,tdim),drag_submits=True, enable_events=True,key='-THUMB3-')],
         #sg.Graph(canvas_size=(tdim,tdim),graph_bottom_left=(0, 0),graph_top_right=(tdim,tdim),drag_submits=True, enable_events=True,key='-THUMB4-'),
         #sg.Graph(canvas_size=(tdim,tdim),graph_bottom_left=(0, 0),graph_top_right=(tdim,tdim),drag_submits=True, enable_events=True,key='-THUMB5-'),
         #sg.Button("Recalage fin",size=(10,1), disabled=True),
        [sg.Column(colonne0),sg.Column(colonne1)],
        [sg.Text(current_version, size=(30, 1),text_color='Tan', font=("Arial", 8, "italic"))],
        ]
window = sg.Window('Open File', layout, location=win_pos,finalize=True)
#window['-FILE1-'].update(WorkDir) 
disp=[]
disp.append(window.Element("-THUMB1-"))
disp.append(window.Element("-THUMB2-"))
disp.append(window.Element("-THUMB3-"))
#disp.append(window.Element("-THUMB4-"))
#disp.append(window.Element("-THUMB5-"))
window.BringToFront()

mosa=window.Element("-MOSA-")


while True:
    event, values = window.read()
    if event==sg.WIN_CLOSED or event=='Cancel': 
        break
    
    if event=='Quit':
        break
    if event=="-FILE1-" :
        
        if values['-FILE1-'] != '' :  
            ImgFiles=values['-FILE1-']
            ImgFiles=ImgFiles.split(';')
            window['Run'].update(disabled=True)
            basefiles=ImgFiles[0].split('.')[0]
            mosa.erase()
            for i in range(len(disp)):
                disp[i].erase()
            
            # faire test si un seul fichier
            if len(ImgFiles) <2 :
                trad("Sélectionner au moins 2 fichiers","Select at least 2 files")
            else :
                
                ImgFile1=ImgFiles[0]
                ImgFile2=ImgFiles[1]
    
                ih=[]
                iw=[]
                centerY=[]
                centerX=[]
                solarR=[]
                y1=[]
                y2=[]
                x1=[]
                x2=[]
                myimg=[]
                a=''
                try :
                    for i in range(len(ImgFiles)) :
                        #lit le fichier fits 1 avec memmap false pour eviter les fichiers ne se ferme pas
                        hdulist1 = fits.open(ImgFiles[i], memmap=False)
                        hdu1=hdulist1[0]
                        mysun1=hdu1.data
                
                        ih.append(hdu1.header['NAXIS2'])
                        iw.append(hdu1.header['NAXIS1'])
                        centerX.append(hdu1.header['CENTER_X'])
                        centerY.append(hdu1.header['CENTER_Y'])
                        solarR.append(hdu1.header['SOLAR_R'])
                        myimg.append(np.reshape(mysun1, (ih[i],iw[i])))
                        print(ImgFiles[i])
                        t=str(centerX[i])+" "+str(centerY[i])+" "+str(solarR[i])
                        trad("Centre x,y - rayon : "+ t, "Center x,y - radius : "+t)
                        print(iw[i],' x ', ih[i])
                        print("")
                        
                        # check des bords 
                        ay1,ay2 = inti.detect_bord(myimg[i], axis=1, offset=0)
                        ax1,ax2 = inti.detect_bord(myimg[i], axis=0, offset=0)
                        y1.append(ay1)
                        y2.append(ay2)
                        x1.append(ax1)
                        x2.append(ax2)
                                                
                        #plt.imshow(myimg)
                        #plt.show()
                        
                        #conversion fits to png8
                        s1=np.percentile(myimg[i], 25)
                        s2=np.percentile(myimg[i],99.9999)*1.05
                        im= fits_to_PIL (myimg[i], s1, s2)
                        
                        # resize pour disp1
                        li=tdim
                        co=tdim
                        imgbytes=img_resize (im,li,co)
                        
                        
                    # affiche en dur des thumbnails TODO en faire une liste
                    #if i in range(len(ImgFiles)) :
                        disp[i].DrawImage(data=imgbytes, location=(0,tdim))
                    
                    # affiche valeurs de rayons
                    by = np.array(y2)-np.array(y1)
                    bx = np.array(x2)-np.array(x1)
                    """
                    byy = [str(int(x)) for x in by]
                    bxx = [str(int(x)) for x in bx]
                    valxy = ', '.join(byy)+' - '+', '.join(bxx)
                    a=[str(x) for x in solarR]
                    a=", ".join(a)
                    window["-RAYONS-"].update(a+' hauteur et largeur disque : '+valxy)
                    """
                    
                    # test si meme dimensions
                    if (np.min(iw)!=np.max(iw)) or (np.min(ih)!=np.max(ih)):
                        trad("Les images n'ont pas la même dimension !!","Images do not have same dimension !!")
                        raise Exception
                    
                    # ordonne les fichiers en partant du plus haut
                    ImgFiles=np.array(ImgFiles)
                    centerY=np.array(centerY)
                    centerX=np.array(centerX)
                    solarR=np.array(solarR)
                    myimg=np.array(myimg)
                    by=np.array(by)
                    bx=np.array(bx)
                    y1=np.array(y1)
                    y2=np.array(y2)
                    x1=np.array(x1)
                    x2=np.array(x2)
                    idx=np.flip(np.argsort(centerY)) #ordre decroissant !
        
                    ImgFiles=ImgFiles[idx]
                    centerY=centerY[idx]
                    centerX=centerX[idx]
                    myimg=myimg[idx]
                    solarR=solarR[idx]
                    by=by[idx]
                    bx=bx[idx]
                    
                    y1=y1[idx]
                    y2=y2[idx]
                    x1=x1[idx]
                    x2=x2[idx]
                    
                    window['Run'].update(disabled=False)
                    #window['Rescale'].update(disabled=False)
                    
                    WorkDir=os.path.dirname(ImgFiles[0])+os.path.sep
                    

                    
                    # resize to value
                    
             
                except :
                    for i in range(len(ImgFiles)) :
                        base=(os.path.basename(ImgFiles[i]))
                        basefich=os.path.splitext(base)[0]
                        ImgFiles[i]=base
                    trad("Probleme de format de fichiers", "Error in files format")
                    trad("Verifier entête fits","Check file header")
                    trad("Mots clefs CENTER_X, CENTER_Y, SOLAR_R", "Keywords CENTER_X, CENTER_Y, SOLAR_R")
                    print(ImgFiles)
                    print("")
                    hdulist1.close()
        else:
            print('file not selected')
        
            
    if event=="Run":
        flag_error=False
        os.chdir(WorkDir)
        for i in range(len(ImgFiles)) :
            base=(os.path.basename(ImgFiles[i]))
            basefich=os.path.splitext(base)[0]
            ImgFiles[i]=base

        
        # nombre de couche de multiresolution
        layers=7
        """
        -------------------------------------------------------------------------
        Creation grande image
        -------------------------------------------------------------------------
        """
        iht=ih[0]
        offset=[]
        
        for i in range(len(ImgFiles)-1):
            iht=iht+ih[i]
            offset.append(centerY[i]-centerY[i+1])
        
        #img_grande=np.zeros((iht,iw[0]))
        #offset=centerY[0]-centerY[1]
            
         
        # correction d'exposition
        for i in range(len(ImgFiles)-1):
            zone_over= y2[i]-offset[i]+y1[i+1]
            print(ImgFiles[i]+" -- "+ImgFiles[i+1])
            t=str( zone_over)+ " pixels"
            trad("Recouvrement de : "+t, "Overlap of : "+t)
            if zone_over <=280:
                if zone_over <=40 :
                    trad("Pas assez de recouvrement !!", "not enought overlap !!")
                    flag_error =True
                else :
                    print("Attention, faible zone de recouvrement, mode degradé ", "Caution, low overlap zone, degraded mode")
                    layers= int(zone_over/40)
                    print("Niveaux de multiresolution : "+ str(layers)," Levels of multiresolution : "+str(layers))
                    
            try :
                z_lum1=np.mean(myimg[i][offset[i]+y1[i+1]:y2[i],:])
                z_lum2=np.mean(myimg[i+1][y1[i+1]:y2[i]-offset[i],:])
                #print("Correction exposition Img2 :", z_lum1/z_lum2)
        
                rlum=z_lum1/z_lum2
                myimg[i+1]=(rlum)*myimg[i+1]
            except:
                pass
    
        if flag_error != True :
            # correction de scaling basée sur le rapport des deux diametres
            
            for i in range(len(ImgFiles)-1):
                sc=solarR[0]/solarR[i+1]
                myimg2_sc=cv2.resize(myimg[i+1],(int(iw[i+1]*sc),int(ih[i+1]*sc)),interpolation=cv2.INTER_LINEAR_EXACT)
                #print('Facteur de rescaling Image'+str(i+1)+' : ',sc)
                
                if sc >=1 :
                    # crop
                    w1=int((int(iw[i+1]*sc)-iw[i+1])/2)
                    w2=-(int(iw[i+1]*sc)-iw[i+1]-w1)
                    h1=int((int(ih[i+1]*sc)-ih[i+1])/2)
                    h2=-(int(ih[i+1]*sc)-ih[i+1]-h1)
                    if h2 == 0 :
                        h2=ih[i]
                    if w2==0 :
                        w2=iw[i]
                    myimg[i+1]=myimg2_sc[h1:h2,w1:w2]
                    #print('shape resized crop :',myimg[i+1].shape)
                else :
                    # padding
                    w1=-int((int(iw[i+1]*sc)-iw[i+1])/2)
                    w2=-(int(iw[i+1]*sc)-iw[i+1]+w1)
                    h1=-int((int(ih[i+1]*sc)-ih[i+1])/2)
                    h2=-(int(ih[i+1]*sc)-ih[i+1]+h1)
                    myimg[i+1][h1:-h2,w1:-w2]=myimg2_sc
                    #print('shape resized padding:',myimg[i+1].shape)
                

            
            img_gr=[]
            #iht=2*ih[0]
            for i in range(len(ImgFiles)):
                # Prepare la grande image
                img_gr.append(np.full((iht,iw[0]),300))
                
            
            # Elimine les bandes noires en bas de l'image 1 et haut de l'image 2
            img_gr[0][:y2[0],:]=myimg[0][:y2[0],:]
            a=0
            for i in range(1,len(ImgFiles)):
                a=a+offset[i-1]
                img_gr[i][a+y1[i]:a+y2[i],:]=myimg[i][y1[i]:y2[i]]
                if i == len(ImgFiles)-1:
                    img_gr[i][a+y1[i]:a+ih[i]-y1[i],:]=myimg[i][y1[i]:ih[i]-y1[i]]
    
            if debug :
                for i in range(len(ImgFiles)) :
                    plt.imshow(img_gr[i])
                    plt.show()
    
            
            
            # gestion de la zone de transition, il faut prendre en compte les artefacts de bords 
            # avec le filtrage gaussien
            a=0
            mid_point=[]
            for i in range(0,len(ImgFiles)-1) :
                a=a+offset[i]
                transition1= (a-offset[i]+y2[i]-(layers)*45)
                transition2=((a+y1[i+1])+(layers)*45)
                m=(transition2-transition1)//2
                m=transition1+m
                mid_point.append(m)
    
            #print(mid_point)
            
            # le coeur de l'algo !!! decomposition en laplacien
            print("")
            trad("Calcul des niveaux de résolution.......", "Resolution levels computation.......")
            gpi=[]
            lpi=[]
            
            for i in range(len(ImgFiles)):
                gpi.append(bl.gaussian_pyramid(img_gr[i], layers))
                lpi.append(bl.laplacian_pyramid(gpi[i]))
         
            # le coeur de l'algo !!! on recompose l'image 
            trad("Recomposition de l'image.......", "Image recomposition.......")
            # on aboute les laplacien avec les transitions definies dans mid_point
            lp_join = bl.laplacian_pyr_join(lpi, mid_point)
            # on recompose l'image car la somme des laplaciens donne l'image finale
            im_join = bl.laplacian_collapse(lp_join)
            
            # on crop au carré - hauteur= largeur de la premiere image
            im_join=im_join[:iw[0], :]
            
            if debug :
                plt.imshow(im_join)
                plt.show()
                
                
            # on recentre et on crop au carré sur la largeur
            im_crop=np.full((iw[0], iw[0]), 300)    # on cree l'image finale
            dy= centerY[0]-iw[0]//2              # offset en y pour recentrage
            dx=centerX[0]-iw[0]//2                  # offset en x pour recentrage
            if iw[0] <= iht :
                if dy<=0 and dx>=0 :
                    im_crop[-dy:iw[0],:]=im_join[0:dy+iw[0], dx:dx+iw[0]]
                
                else :
                    trad("Allo Houston, on n'a un problème","Allo houston, we have a problem..")
                    im_crop=[]
                    im_crop=np.copy(im_join)
            else :
                dec=(iw[0]-iht)//2
                if dy<=0 and dx>=0 :
                    im_crop[-dy:iht-dy,:]=im_join[0:iht, dx:dx+iw[0]]
                
                else :
                    trad("Allo Houston, on n'a un problème","Allo houston, we have a problem..")
                    im_crop=[]
                    im_crop=np.copy(im_join)
                
                
            
            #conversion fits to png8
            s1=np.percentile(im_crop, 25)
            s2=np.percentile(im_crop,99.9999)*1.05
            im= fits_to_PIL (im_crop, s1, s2)
           
            # resize pour display
            li=gdim
            co=gdim
            imgbytes=img_resize (im,li,co)
            
            # affiche 
            mosa.DrawImage(data=imgbytes, location=(0,gdim))

            
            # sauve fits
            hdu1.header['NAXIS2']=iw[0]
            hdu1.header['centerY']=centerY[0]
            hdu1.header['solar_R']=solarR[0]
            nom_fits_short=ImgFiles[0].split('.')[0]+"-"+ImgFiles[len(ImgFiles)-1].split('.')[0]+".fits"
            nom_fits=WorkDir+nom_fits_short
            DiskHDU=fits.PrimaryHDU(im_crop,hdu1.header)
            DiskHDU.writeto(nom_fits, overwrite='True')
            
            # sauve png
            #im_join=np.array(im_join, dtype='uint16')
            s2=np.percentile(im_crop, 25)
            s1=np.percentile(im_crop,99.9999)*1.05
            im = inti.seuil_image_force(im_crop,s1,s2)
            im_png=np.array(im, dtype='uint16')
            im_png=cv2.flip(im_png,0)
            nom_png_short=ImgFiles[0].split('.')[0]+"-"+ImgFiles[len(ImgFiles)-1].split('.')[0]+".png"
            nom_png=WorkDir+nom_png_short
            cv2.imwrite(nom_png,im_png)
            
            
            print("")
            trad("Succès !!", "Success !!")
            trad("Image fits : "+nom_fits_short, "Fits image : "+nom_fits_short)
            trad("Image png : "+nom_png_short, "Png image : "+nom_png_short)
            print("")
                
            
    """       
    if event=="Rescale":
        flag_error=False
        os.chdir(WorkDir)
        for i in range(len(ImgFiles)) :
            base=(os.path.basename(ImgFiles[i]))
            basefich=os.path.splitext(base)[0]
            ImgFiles[i]=base
   
        
        if flag_error != True :
            # correction de scaling basée sur le rapport des deux diametres
            
            # valeur de reference
            #solarR_ref=int(values["-RAYREF-"])
            bx_ref=int(np.min(bx))
            by_ref=int(np.min(by))
            solarR_ref=bx_ref//2
            
             

        for i in range(len(ImgFiles)):
            #sc=solarR_ref/solarR[i]
            scx=bx_ref/bx[i]
            scy=by_ref/by[i]
            #scx=1
            #scy=1
            myimgR=np.copy(myimg[i])
            
            myimg2_sc=cv2.resize(myimg[i],(int(iw[i]*scx),int(ih[i]*scy)),interpolation=cv2.INTER_LINEAR_EXACT)
            print('Facteur de rescaling Image R en largeur et hauteur'+str(i+1)+' : ',scx, scy) 
            print(myimg2_sc.shape)
            
            w1=int((int(iw[i]*scx)-iw[i])/2)
            w2=-(int(iw[i]*scx)-iw[i]-w1)
            h1=int((int(ih[i]*scy)-ih[i])/2)
            h2=-(int(ih[i]*scy)-ih[i]-h1)
            if w2==0 :
                w2=iw[i]
            if h2==0 :
                h2=ih[i]
            print(w1,w2,h1,h2)
            
            
            if scx >=1 and scy >=1:
                # crop en largeur et hauteur
                myimg[i]=myimg2_sc[h1:h2,w1:w2]
                print('shape resized crop :',myimg[i].shape)
            
            if scx<1 and scy<1 :
                myimg[i][-h1:-h2,-w1:-w2]=myimg2_sc
                print('shape resized padding:',myimg[i].shape)
                
            if scy>= 1 and scy <1:
                myimg[i][:,-w1:w2]=myimg2_sc[h1:h2,:]
                print('shape resized y crop x padd:',myimg[i].shape)
            
            if scx<1 and scy>=1:
                myimg[i][-h1:h2,:]=myimg2_sc[:,w1:w2]
                print('shape resized y padd x crop:',myimg[i].shape)

            
     
        # On sauve
        for i in range(len(ImgFiles)):
            # sauve fits
            hdu1.header['NAXIS2']=ih[i]
            hdu1.header['centerY']=centerY[i]
            hdu1.header['solar_R']=solarR_ref
            nom_fits_short=ImgFiles[i].split('.')[0]+"_r.fits"
            nom_fits=WorkDir+nom_fits_short
            DiskHDU=fits.PrimaryHDU(myimg[i],hdu1.header)
            DiskHDU.writeto(nom_fits, overwrite='True')
            
            # sauve png
            #im_join=np.array(im_join, dtype='uint16')
            im_join=myimg[i]
            s2=np.percentile(im_join, 25)
            s1=np.percentile(im_join,99.9999)*1.05
            im = inti.seuil_image_force(im_join,s1,s2)
            im_png=np.array(im, dtype='uint16')
            im_png=cv2.flip(im_png,0)
            nom_png_short=ImgFiles[i].split('.')[0]+"_r.png"
            nom_png=WorkDir+nom_png_short
            cv2.imwrite(nom_png,im_png)
            print("Images rescaled", nom_fits_short, nom_png)
    """
        
window.close()   


#met a jour le repertoire si on a changé
try:
    my_dictini['obs_dir']=WorkDir
    #print(WorkDir)
    hdulist1.close()
    
    with open(my_ini, "w") as f1:
        yaml.dump(my_dictini, f1, sort_keys=False)
except:
    pass



