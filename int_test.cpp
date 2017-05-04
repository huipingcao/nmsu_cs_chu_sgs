/* floor example */
#include <stdio.h>      /* printf */
#include <string.h>      /* printf */
#include <math.h>       /* floor */

int main ()
{
    int i = floor(11*0.6);
    printf ( "%d\n",  i);

    char str[80];
    strcpy (str,"these ");
    strcat (str,"strings ");
    strcat (str,"are ");
    strcat (str,"concatenated.");
    puts (str);
    char *cp = new char[1000];
    strcpy(cp, str);
    puts (strcat(str, " hello"));
    puts (str);
    puts (cp);

    int t = 20;
    int m = ++t;
    printf("%d\n", m);
    printf("%d\n", m);
}
