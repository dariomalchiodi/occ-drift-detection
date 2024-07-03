library(stringr)


z_alfa_mezzi_upper=function(alfa){qnorm(alfa/2,lower.tail=F)}
z_alfa=function(alfa){qnorm(alfa)}


################## SOLO SIGNIFICATIVITà ####################################

sample_size_alfa=function(alfa, d){ ceiling(z_alfa(1-alfa/2)^2*1/(2*d^2))}
# d: distanza da rilevare (effect size)

detected_distance_alfa=function(alfa,n){z_alfa(1-alfa/2)/sqrt(2*n)}
#n: sample size



plot_alfa_samplesizeVSd=function(d_range, nmax, alfa){
    #il primo valore di alfa è quello di riferimento che verrà colorato di blu, 
    # per gli altri valori di alfa usata tavolozza dal giallo al rosso

    n_plots=length(alfa)-1
    if(n_plots>0){
        title=paste("color : towards yellow for decreasing significance \n alpha = ", 
                             toString(sort(alfa)),"\n blue for alpha = ",
                             toString(alfa[1]))
        }
        else{title= paste("alpha =", as.character(alfa[1]))}
                     
    curve(sample_size_alfa(alfa[1],x),xlim=d_range,
      ylim=c(0,nmax), xlab="detected difference", ylab="sample size", 
          main=title,
          col="blue", cex.axis=1.5, cex.lab=1.5)
    
    if(n_plots>0){
        c=sort(colorRampPalette(c("gold1", "red"))(n_plots), decreasing=F)
        
        a=sort(alfa[2:length(alfa)])
        for(k in seq(1,n_plots)){
            curve(sample_size_alfa(a[k],x),
                  xlim=d_range, ylim=c(0,nmax) ,
                  col=c[k], cex.axis=1.5, cex.lab=1.5, add=T)
            
            }
        }
}


plot_alfa_effectsizeVSn=function(n_range, dmax, alfa){
#il primo valore di alfa è quello di riferimento che verrà colorato di blu, 
    # per gli altri valori di alfa usata tavolozza dal giallo al verde

    n_plots=length(alfa)-1
    if(n_plots>0){
        title=paste("color : towards yellow for decreasing significance \n alpha = ", 
                             toString(sort(alfa)),"\n blue for alpha = ",
                             toString(alfa[1]))
        }
        else{title= paste("alpha =", as.character(alfa[1]))}
    
    curve(detected_distance_alfa(alfa[1],x),
          ylim=c(0,dmax),xlim=n_range,
           main=title,
          xlab="sample size" , ylab="detected difference",col="blue", 
          cex.axis=1.5, cex.lab=1.5)


    n_plots=length(alfa)-1
    if(n_plots>0){
        c=sort(colorRampPalette(c("gold1", "chartreuse4"))(n_plots), decreasing=F)
        a=sort(alfa[2:length(alfa)])
        for(k in seq(1,n_plots)){
            curve(detected_distance_alfa(a[k],x),
                  ylim=c(0,dmax),xlim=n_range,
                 col=c[k], cex.axis=1.5, cex.lab=1.5, add=T)
            
            }
        }
}

################## SIGNIFICATIVITà e POTENZA ####################################

sample_size_alfa_beta=function(alfa,beta, d){ceiling(1/2*((z_alfa(1-alfa/2)+z_alfa(1-beta))/d)^2)}
# z_alfa: z alfa mezzi upper
# z_beta: z beta upper
# d: distanza da rilevare (effect size)

#detected_distance_beta=function(z_alfa,z_beta,n){(z_alfa+z_beta)/sqrt(2*n)}
detected_distance_beta=function(alfa,potenza,n){
    (z_alfa(1-alfa/2)+z_alfa(potenza))/sqrt(2*n)}


plot_beta_samplesizeVSd=function(d_range, nmax, alfa, potenza){
    beta=1-potenza
    n_plots=length(potenza)-1
    if(n_plots>0){
        title=paste("color : towards yellow for decreasing power \n ", 
                             toString(sort(potenza, decreasing=T)),"\n blue for power = ",
                             toString(potenza[1]))
        }
        else{title= paste("power =", as.character(potenza[1]))}
    
    curve(sample_size_alfa_beta(alfa,beta[1],x),
          xlim=d_range,ylim=c(0,nmax), 
          main=title,
          xlab=paste("detected difference, alfa = ", as.character(alfa)), ylab="sample size",
          col="blue", cex.axis=1.5, cex.lab=1.5)
    
    if(n_plots>0){
        c=sort(colorRampPalette(c("gold1", "red"))(n_plots), decreasing=F)
        a=sort(beta[2:length(beta)])
        for(k in seq(1,n_plots)){
            curve(sample_size_alfa_beta(alfa,a[k],x),
                  xlim=d_range,ylim=c(0,nmax),
                  col=c[k], cex.axis=1.5, cex.lab=1.5, add=T)


            }
        }
}

#####
plot_beta_effectsizeVSn=function(n_range, dmax, alfa, potenza){

    beta=1-potenza
    n_plots=length(potenza)-1
    if(n_plots>0){
        title=paste("color : towards yellow for decreasing power \n ", 
                             toString(sort(potenza, decreasing=T)),"\n blue for power = ",
                             toString(potenza[1]))
        }
        else{title= paste("power =", as.character(potenza[1]))}
    

    curve(detected_distance_beta(alfa,potenza[1],x),
          xlim=n_range,ylim=c(0,dmax),
          main=title,
          xlab= paste("sample size, alpha = ", as.character(alfa)), 
          ylab="detected difference ",col="blue")

    if(n_plots>0){
        c=sort(colorRampPalette(c("gold1", "chartreuse4"))(n_plots), decreasing=F)
        a=sort(potenza[2:length(potenza)])
        for(k in seq(1,n_plots)){
            curve(detected_distance_beta(alfa,a[k],x),
                  xlim=n_range,ylim=c(0,dmax),
                  col=c[k], cex.axis=1.5, cex.lab=1.5, add=T)
            }
        }
}

##########################################################################

power=function(d,alfa,n){
    z1=qnorm(1-alfa/2)
    z2=qnorm(alfa/2)
    return(1-pnorm(z1-d*sqrt(2*n))+pnorm(z2-d*sqrt(2*n)))}

plot_power_alfa=function(n, alfa, xlim){
     xlim=xlim

    n_plots=length(alfa)-1
    print(n_plots)
    
    if(n_plots>0){
        title=paste("color : towards yellow for decreasing signnificance \n alpha = ", 
                             toString(sort(alfa)),"\n blue for alpha = ",
                             toString(alfa[1]))
        }
        else{title= paste("alpha =", as.character(alfa[1]))}
    
    curve(power(x,alfa[1],n),xlim=xlim, 
          main=title,
          xlab=paste("detected difference, n = ", as.character(n)),
          ylab="power",col="blue", cex.axis=1.5, cex.lab=1.5)


    if(n_plots>0){
        c=sort(colorRampPalette(c("gold1", "red"))(n_plots), decreasing=F)
        a=sort(alfa[2:length(alfa)])
        for(k in seq(1,n_plots)){
            curve(power(x,a[k],n),xlim=xlim, 
                  col=c[k], cex.axis=1.5, cex.lab=1.5, add=T)

            }
        }
}


plot_power_samplesize=function(samplesize, alfa, xlim){
     xlim=xlim

    n_plots=length(samplesize)-1
    
    if(n_plots>0){
        title=paste("color : towards yellow for decreasing sample size \n n = ",
                    toString(sort(samplesize, decreasing=T)), "\n blue n = ", 
                    as.character(samplesize[1] ))
        }
        else{title= paste("n =", as.character(samplesize[1]))}
    
    curve(power(x,alfa,samplesize[1]),xlim=xlim, 
          main=title,
          xlab=paste("detected difference, alpha = ", as.character(alfa)),
          ylab="power",col="blue", cex.axis=1.5, cex.lab=1.5)


    if(n_plots>0){
        c=sort(colorRampPalette(c("gold1", "red"))(n_plots),decreasing=T)
        a=sort(samplesize[2:length(samplesize)])
        for(k in seq(1,n_plots)){
            curve(power(x,alfa,a[k]),xlim=xlim, 
                  col=c[k], cex.axis=1.5, cex.lab=1.5, add=T)

            }
        }
}

