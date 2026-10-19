rm(list=ls())
#----------------------- INSTALACIÓN DE PAQUETES -------------------------------
devtools::install_github("JaseZiv/worldfootballR")
library(worldfootballR)
library(purrr)
library(dplyr)
library(lubridate)

#--------------------------- FUNCIONES AUXILIARES ------------------------------

#Cambia el formato de la temporada de "17/18" a "2017"
cambia_formato_temporada <- function(x) {
  year <- as.numeric(substr(x, 1, 2))
  ifelse(year >= 70, 1900 + year, 2000 + year)
}

#Calcula la edad de alguien nacido en "nacimiento" en el dia "dia". El formato
#de las fechas es: "YYYY-MM-DD"
calcula_edad <- function(nacimiento,dia){
  floor(time_length(interval(nacimiento, dia), "years"))
}

#Comprueba que la funcion tm_player_transfer_history funcione 
#(Devuelve TRUE si el link del jugador dará error y FALSE si no)
error_tm_player_trasnfer_history <- function(link){
  prueba <- safely(tm_player_transfer_history)
  return(!is.null(prueba(link)$error))
}


# Jugadores que dan error al usar la funcion tm_player_transfer_history
error_link_jugadores <- c(
  "https://www.transfermarkt.com/miguel-angel-garcia/profil/spieler/854402",
  "https://www.transfermarkt.com/luis-david-rodriguez/profil/spieler/1000734"
)

#---------------------------- CARGA DE DATOS -----------------------------------
temporadas <- 2026:2026
liga <- c('https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1',
          'https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES1',
          'https://www.transfermarkt.com/SERIEA/startseite/wettbewerb/IT1',
          'https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/L1',
          'https://www.transfermarkt.com/ligue1/startseite/wettbewerb/FR1'
          )

if (length(temporadas) > 1 | length(liga) > 1){
  valores <- data.frame()
  for (i in temporadas) {
    for (j in liga){
    print(paste0("Scrapeando temporada ", i, " de ", j))
    df <- tm_player_market_values(start_year = i, league_url = j)
    df <- df[,c(3,4,5,7,8,9,10,11,13,14,18,19)]
    if (any(is.na(df$player_market_value_euro))){
      df <- df[-which(is.na(df$player_market_value_euro)), ]
    }
    valores <- rbind(valores,df)
  }}
} else {
  print(paste0("Scrapeando temporada ", temporadas, " de ", liga))
  valores <- tm_player_market_values(start_year = temporadas, league_url = liga)
  valores <- valores[,c(3,4,5,7,8,9,10,11,13,14,18,19)]
  valores <- valores[-which(is.na(df$player_market_value_euro)), ]
}

#Una vez tenemos la tabla, la descargamos con el siguiente comando

write.csv(valores, "ruta deseada", row.names = FALSE)
