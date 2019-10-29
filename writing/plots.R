# plots for showing the patterns in the data

library(dplyr)
library(data.table)
library(ggplot2)
# library(manipulate)

setwd('~/git/lidc-representation/writing')
df_raw <- fread('simdata.csv')
table <-function(...) base::table(..., useNA = 'always')

# sample only 100 observations for better visualization
nobs = nrow(df_raw)
# nobs = 1000
df <- df_raw[1:nobs]
df[, t:=factor(t)]

df[, x_quant:=cut(x, breaks = quantile(x, probs = seq(0, 1, length.out = 5)), include.lowest = T)]
df[, z_quant:=cut(z, breaks = quantile(z, probs = seq(0, 1, length.out = 11)), include.lowest = T)]

# set some xjust values for plotting
xjust=-.56
xscale=4.48

df[, x_quant:=factor(x_quant, labels=c('smallest', 'small', 'large', 'largest'))]
dfmarginal <- df[, list(y=mean(y), grp=1), by = 't']
dfxquant   <- df[, list(y=mean(y)), by=c('t', 'x_quant')]
dfxquant[, xpos:=xjust+as.numeric(t)+as.numeric(x_quant) / xscale]
df[dfxquant, xpos:=i.xpos, on=c('x_quant', 't'), by =.EACHI]

# xquantcolors <- scales::hue_pal()(4)
xquantcolors <- rev(RColorBrewer::brewer.pal(4, "RdYlBu"))
xmargcolor   <- RColorBrewer::brewer.pal(8, "Dark2")[8]
colors <- c(xmargcolor, xquantcolors)
rbindlist(list(
  'a: Marginal'=df %>%mutate(x_quant='(any)'),
  'b: Conditional'=df
), idcol='dftype') %>%
  ggplot(aes(x=u1, y=u2, col=x_quant)) + 
  geom_point(alpha=.5) +
  facet_grid(.~dftype) + 
  scale_color_manual(values=colors) +
  labs(col='x (tumor size) quartile') +
  theme_minimal() ->  pu

width=13.6
height=7.47
sizefactor = 2
width = width / sizefactor; height=height / sizefactor; dpi=300
ggsave('plotu1u2.png', pu, width=width, height=height, dpi=dpi)





# manipulate({

g <- df %>%
  ggplot(aes(x=t, y=y, col=x_quant)) + 
  geom_violin(adjust=5, trim=F) +
  geom_jitter(width=.1, height=0, alpha=.1, aes(x=xpos)) +
  geom_violin(aes(col=NULL), alpha=.1, adjust=5, trim=F) +
  geom_line(data=dfmarginal, aes(group=grp, col=NULL), size=1) +
  geom_point(data=dfmarginal, aes(col=NULL), size=2) +
  geom_line(data=dfxquant,   aes(group=x_quant, x=xpos)) +
  geom_point(data=dfxquant,  aes(x=xpos), size=1.5) +
  theme_minimal() + 
  scale_color_manual(values=xquantcolors) +
  labs(col='x (tumor size) quartile', x='treatment', y = 'y (survival)')

# g

ggsave('dataplot1.png', g, width = width, height=height, dpi=dpi)

# wrong measurements
df[, xreal:=x+5]
df[, xpow3:=xreal^3]
df[, xpow1div3:=xreal^(1/3)]

rbindlist(list(
  'a: cube'=df[, list(xreal, xmeas=xpow3)],
  'b: invcube'=df[, list(xreal, xmeas=xpow1div3)]
), idcol='x_measurement') %>%
  ggplot(aes(x=xmeas, y=xreal)) + 
  geom_point() +
  facet_grid(.~x_measurement, scales = 'free_x') +
  theme_minimal() + 
  labs(x='Measured x (tumor size)', y = 'True x (tumor size)') -> pxmeas

# pxmeas
ggsave('xmeas.png', pxmeas, width=width, height=height, dpi=dpi)

# }, 
# xjust=slider(-.6,-0.5, step = .01), xscale=slider(4,5, step=.01)
# )
